"""vLLM 异步 rollout：OpenAI 兼容客户端 + LoRA 热同步。

为什么单独一个文件
------------------
训练循环（``train_grpo.py``）现在直接持有 HF 模型做采样；换成 vLLM 之后，
采样走 HTTP，训练仍在本进程算 logp/梯度。这里只提供两样东西：

1. :class:`VLLMChatClient` —— 长得像 ``HFPolicyClient``，只需要 ``.chat()``；
2. :class:`LoRASyncer` —— 把训练进程刚更新的 LoRA 推进 vLLM（每步或每 N 步）。

权重同步为什么用 LoRA 热加载
---------------------------
vLLM 启动时 ``--enable-lora``，之后可以用 HTTP 动态装卸 adapter：

    POST /v1/load_lora_adapter   {"lora_name": "...", "lora_path": "..."}
    POST /v1/unload_lora_adapter {"lora_name": "..."}

比“每步重启引擎”（约 84s）便宜得多；比 ``update_weights`` 的分布式 RPC 简单得多。
代价是每步要把 adapter（本项目 142MB）落一次盘。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class HTTPUsage:
    """与 ``rollout.llm_client.Usage`` 同构，但只统计 token（本地服务不计费）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, p: int, c: int) -> None:
        with self._lock:
            self.prompt_tokens += int(p)
            self.completion_tokens += int(c)
            self.calls += 1

    @property
    def cost_cny(self) -> float:
        return 0.0

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "calls": self.calls, "cost_cny": 0.0}


class VLLMChatClient:
    """把 vLLM 的 OpenAI 兼容接口包装成 ``.chat(messages)``。"""

    def __init__(self, base_url: str, model: str, *, temperature: float = 1.0,
                 max_tokens: int = 256, api_key: str = "EMPTY", timeout: float = 120.0):
        from openai import OpenAI

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = model
        self.usage = HTTPUsage()
        self._client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)

    @property
    def model_name(self) -> str:
        return self.name

    def set_model(self, name: str) -> None:
        self.model = name
        self.name = name

    def chat(self, messages: list[dict], **_: object) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        u = resp.usage
        if u:
            self.usage.add(u.prompt_tokens or 0, u.completion_tokens or 0)
        return text


class LoRASyncer:
    """把本地 adapter 目录热加载进 vLLM，并切到新名字。"""

    def __init__(self, base_url: str, *, api_key: str = "EMPTY", timeout: float = 180.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.current: str | None = None
        self.history: list[str] = []

    # ---------------- HTTP ----------------

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def load(self, name: str, path: str) -> dict:
        return self._post("/load_lora_adapter", {"lora_name": name, "lora_path": path})

    def unload(self, name: str) -> dict:
        return self._post("/unload_lora_adapter", {"lora_name": name})

    # ---------------- 高层 ----------------

    def push(self, name: str, path: str) -> float:
        """加载新 adapter 并卸掉旧的；返回本次同步耗时（秒）。"""
        t0 = time.perf_counter()
        self.load(name, path)
        if self.current and self.current != name:
            try:
                self.unload(self.current)
            except (urllib.error.URLError, urllib.error.HTTPError):
                # 卸载失败不影响正确性：旧 adapter 只是占一个 LoRA 槽
                pass
        self.current = name
        self.history.append(name)
        return time.perf_counter() - t0


def check_server(base_url: str, timeout: float = 5.0) -> dict:
    """确认服务活着，返回 /v1/models（base_url 是 OpenAI 风格，含 /v1）。"""
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))
