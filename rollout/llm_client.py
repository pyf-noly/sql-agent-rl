"""OpenAI 兼容 API 的最小封装：重试、并发、用量统计、磁盘缓存。

为什么要这么封装
----------------
1. **可切换**：本地 vLLM/SGLang 起服务后也是 OpenAI 兼容接口，
   只改 ``base_url`` 就能从"调 API"切到"跑自己的模型"，业务代码一行不动；
2. **可重放**：每条请求按 (model, messages, temperature, seed) 哈希缓存，
   调试脚本时不重复花钱；缓存命中率也能反映实验设计的复用度；
3. **可计费**：累计 prompt/completion token，按单价折算成人民币，
   方便回答"这个实验花了多少钱"。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

#: 单价（元 / 1M tokens），按需改。默认按 DeepSeek 公开价量级估。
PRICE_INPUT_PER_M = 2.0
PRICE_OUTPUT_PER_M = 8.0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, p: int, c: int) -> None:
        with self._lock:
            self.prompt_tokens += p
            self.completion_tokens += c
            self.calls += 1

    def add_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def add_retry(self) -> None:
        with self._lock:
            self.retries += 1

    @property
    def cost_cny(self) -> float:
        return (self.prompt_tokens / 1e6 * PRICE_INPUT_PER_M
                + self.completion_tokens / 1e6 * PRICE_OUTPUT_PER_M)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "cost_cny": round(self.cost_cny, 4),
        }


class LLMClient:
    """线程安全的 OpenAI 兼容客户端，带磁盘缓存。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        cache_dir: str | Path = ".cache/llm",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 5,
    ):
        self.base_url = base_url or os.environ["OPENAI_BASE_URL"]
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.model = model or os.environ["OPENAI_MODEL"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120.0)
        self.usage = Usage()

    # ---------------- 缓存 ----------------

    def _cache_key(self, messages: list[dict], extra: dict) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **extra,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ---------------- 调用 ----------------

    def chat(self, messages: list[dict], *, use_cache: bool = True, **extra: Any) -> str:
        """返回模型输出的文本。失败时抛异常（调用方决定怎么记）。"""
        key = self._cache_key(messages, extra)
        path = self._cache_path(key)
        if use_cache and path.is_file():
            self.usage.add_hit()
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                u = resp.usage
                if u:
                    self.usage.add(u.prompt_tokens, u.completion_tokens)
                if use_cache:
                    path.write_text(
                        json.dumps({"text": text, "model": self.model}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.usage.add_retry()
                if attempt == self.max_retries:
                    break
                sleep = min(2 ** attempt, 30) + random.random()
                print(f"\n      [重试 {attempt}/{self.max_retries}] {type(e).__name__}: "
                      f"{str(e)[:120]}，{sleep:.1f}s 后重试")
                time.sleep(sleep)
        raise RuntimeError(f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_err}")


def make_client_from_env(**kw: Any) -> LLMClient:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    return LLMClient(**kw)
