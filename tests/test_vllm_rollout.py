"""vLLM rollout 客户端的协议测试：用本地假 HTTP 服务验证 URL / JSON 形状。

不需要真的 vLLM、不需要 GPU。测三件事：
1. ``check_server`` 能读到 /v1/models；
2. ``LoRASyncer.push`` 会 POST load 新 adapter、再 POST unload 旧的；
3. ``VLLMChatClient.chat`` 发的是 OpenAI 兼容请求，能拿到 content 与 usage。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from training.vllm_rollout import LoRASyncer, VLLMChatClient, check_server


class _Handler(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict]] = []
    model = "sft"

    def log_message(self, *args):  # 安静
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "sft"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        type(self).calls.append((self.path, payload))
        if self.path == "/v1/chat/completions":
            self._send({
                "id": "x", "object": "chat.completion", "model": payload.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": "```sql\nSELECT 1\n```"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            })
        elif self.path in ("/v1/load_lora_adapter", "/v1/unload_lora_adapter"):
            self._send({"status": "ok"})
        else:
            self._send({"error": "not found"}, 404)


@pytest.fixture()
def fake_server():
    _Handler.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    yield base
    srv.shutdown()


def test_check_server(fake_server):
    info = check_server(fake_server + "/v1")
    assert info["data"][0]["id"] == "sft"


def test_lora_push_loads_then_unloads_previous(fake_server):
    syncer = LoRASyncer(fake_server + "/v1")
    syncer.push("grpo_s1", "/tmp/adapter1")
    syncer.push("grpo_s2", "/tmp/adapter2")
    loads = [p for path, p in _Handler.calls if path == "/v1/load_lora_adapter"]
    unloads = [p for path, p in _Handler.calls if path == "/v1/unload_lora_adapter"]
    assert loads[0] == {"lora_name": "grpo_s1", "lora_path": "/tmp/adapter1"}
    assert loads[1] == {"lora_name": "grpo_s2", "lora_path": "/tmp/adapter2"}
    # 第二次 push 应把第一个卸掉
    assert unloads == [{"lora_name": "grpo_s1"}]
    assert syncer.current == "grpo_s2"


def test_chat_client_posts_openai_payload(fake_server):
    client = VLLMChatClient(fake_server + "/v1", "sft", temperature=0.0, max_tokens=32)
    text = client.chat([{"role": "user", "content": "hi"}])
    assert "SELECT 1" in text
    paths = [p for p, _ in _Handler.calls]
    assert "/v1/chat/completions" in paths
    payload = dict(_Handler.calls)["/v1/chat/completions"]
    assert payload["model"] == "sft"
    assert payload["max_tokens"] == 32
    assert client.usage.to_dict()["prompt_tokens"] == 11
