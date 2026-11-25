#!/usr/bin/env python3
"""验证 vLLM 服务与 HF 训练端的 prompt 是否**逐 token 一致**。

为什么重要：训练时 HF 端用 ``training.chatfmt.apply_template`` 构造序列；
vLLM 端用自己的 chat template。如果两者不一致，异步 rollout 学的是另一种输入分布，
A/B 的结论就不成立。这个脚本拿 5 条 BIRD 样本对比 token id。

用法::

    python scripts/check_vllm_template.py --base-url http://127.0.0.1:8000/v1 --model sft
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.prompt import BIRD_PROMPT, build_messages  # noqa: E402
from training.chatfmt import apply_template  # noqa: E402
from training.model import load_tokenizer  # noqa: E402


def vllm_tokenize(base_url: str, model: str, messages: list[dict]) -> list[int]:
    # vLLM 的 /tokenize 挂在应用根路径，不在 /v1 下
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    payload = json.dumps({"model": model, "messages": messages,
                          "add_generation_prompt": True}).encode("utf-8")
    req = urllib.request.Request(
        root + "/tokenize", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d.get("tokens") or d.get("input_ids") or []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="sft")
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/qwen3-4b")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    reg = DatabaseRegistry(ROOT / "data/raw/bird/dev_20240627/dev_databases")
    rows = json.loads((ROOT / "data/bird/dev.json").read_text(encoding="utf-8"))
    bad = 0
    for i, r in enumerate(rows[: args.n]):
        schema = reg.schema(r["db_id"])
        messages = build_messages(r["question"], schema, BIRD_PROMPT,
                                  evidence=r.get("evidence"))
        hf_ids = apply_template(tok, messages, add_generation_prompt=True)
        v_ids = vllm_tokenize(args.base_url, args.model, messages)
        same = hf_ids == v_ids
        if not same:
            bad += 1
            k = next((j for j in range(min(len(hf_ids), len(v_ids)))
                      if hf_ids[j] != v_ids[j]), None)
            print(f"[{i}] MISMATCH len hf={len(hf_ids)} vllm={len(v_ids)} "
                  f"first_diff={k}")
            if k is not None:
                print("   hf  :", tok.decode(hf_ids[max(0, k - 20):k + 20]))
                print("   vllm:", tok.decode(v_ids[max(0, k - 20):k + 20]))
        else:
            print(f"[{i}] OK  len={len(hf_ids)}")
    print(f"\n一致 {args.n - bad}/{args.n}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
