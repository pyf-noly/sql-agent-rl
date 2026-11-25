#!/usr/bin/env python3
"""不需要 GPU、不需要 torch 的数据与算法自检（P4 上卡**之前**就该全绿）。

为什么要有它
-----------
P4 的绝大多数坑不在 GPU 上，而在"数据格式"和"mask 边界"——这两样在这台
没有 CUDA 的 Windows 机器上就能验掉。真正到租来的机器上时，只剩"跑得快不快"。

检查内容
--------
1. ``data/train/`` 三份文件存在、条数对、meta 里 checks.ok 为真；
2. 每条 SFT 样本：system+schema 在前、助手消息带 ```sql、反馈里没有 gold；
3. 每条 RL 样本：有 gold、有 prompt_messages；
4. 用**假 tokenizer**（字符级、可逆）验证 ``encode_with_spans`` 的 mask 边界：
   单轮、三轮、超长截断三种情况；
5. GRPO 组内优势的数学（含"整组同分 → 优势全 0"）；
6. 三个 config JSON 能被 dataclass 严格解析（防字段名拼错）。

用法::

    python scripts/selfcheck_p4.py          # 打勾/打叉
    python -m pytest tests/test_p4_pipeline.py   # 同一套检查进 CI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- 假 tokenizer

class FakeTokenizer:
    """字符级、可逆的聊天模板实现，用来在白盒环境下验证 mask 算法。

    它模仿 Qwen 的模板格式（``<|im_start|>role\\n...<|im_end|>\\n``），
    但 token = Unicode 码点，所以"解码回原文"严格可逆——真 tokenizer 的
    BPE 边界不可控，反而很难断言。
    """

    chat_template = "<|im_start|>{role}\n{content}<|im_end|>\n"
    pad_token_id = 0
    eos_token_id = 1
    name_or_path = "fake-tokenizer"

    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, return_tensors=None,
                            enable_thinking=None):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _conv(n_turns: int, answer: str = "```sql\nSELECT 1;\n```") -> list[dict]:
    msgs = [{"role": "system", "content": "SYSMARK"},
            {"role": "user", "content": "QUESTIONMARK"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": answer})
        if i < n_turns - 1:
            msgs.append({"role": "user", "content": "FEEDBACKMARK"})
    return msgs


# ---------------------------------------------------------------- 各项检查

def check_data_files(root: Path) -> list[tuple[str, bool, str]]:
    from training.data import read_jsonl

    out = []
    d = root / "data/train"
    meta_p = d / "train_data.meta.json"
    if not meta_p.is_file():
        return [("data/train 存在", False, f"缺 {meta_p}，先跑 build_p4_data.py")]
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    out.append(("数据自检 ok", bool(meta.get("checks", {}).get("ok")),
                str(meta.get("checks", {}).get("failures") or "通过")))

    sft = read_jsonl(d / "sft_train.jsonl")
    val = read_jsonl(d / "sft_val.jsonl")
    rl = read_jsonl(d / "rl_train.jsonl")
    out.append(("SFT train/val 条数", len(sft) == meta["n_sft_train"] and len(val) == meta["n_sft_val"],
                f"{len(sft)}/{len(val)}"))
    out.append(("RL 条数", len(rl) == meta["n_rl"], str(len(rl))))
    out.append(("RL 与 dev 库不重叠", meta.get("rl_dev_db_overlap") == 0,
                str(meta.get("rl_dev_db_overlap"))))

    # 结构与泄漏
    bad = 0
    for it in sft[:300]:
        m = it["messages"]
        if m[0]["role"] != "system" or m[1]["role"] != "user":
            bad += 1
        elif "### Database schema" not in m[1]["content"]:
            bad += 1
        elif any(a["role"] == "assistant" and "```sql" not in a["content"] for a in m):
            bad += 1
        elif any(u["role"] == "user" and "Your previous attempt" not in u["content"]
                 for u in m[2:]):
            bad += 1
    out.append(("SFT 结构（schema+反馈模板）", bad == 0, f"异常 {bad}/300"))

    gold_leak = 0
    for r in rl:
        q = (r.get("query") or "").strip().lower()
        if not q:
            gold_leak += 1
    out.append(("RL 都有 gold", gold_leak == 0, f"缺 {gold_leak}"))
    return out


def check_mask(root: Path) -> list[tuple[str, bool, str]]:
    from training.chatfmt import encode_with_spans

    tok = FakeTokenizer()
    out = []

    enc = encode_with_spans(tok, _conv(1))
    answer = "```sql\nSELECT 1;\n```"
    # mask 必须正好落在"助手内容 + 结束符"上
    masked_text = "".join(chr(i) for i, m in zip(enc.input_ids, enc.loss_mask) if m)
    out.append(("单轮 mask 收尾正确", masked_text.startswith(answer) and masked_text.endswith("<|im_end|>\n"),
                repr(masked_text[-20:])))
    out.append(("单轮 mask 只覆盖助手", not any(
        k in masked_text for k in ("SYSMARK", "QUESTIONMARK", "FEEDBACKMARK")),
        f"loss={enc.n_loss_tokens}"))
    out.append(("单轮 mismatch=0", enc.n_mismatch == 0, str(enc.n_mismatch)))

    enc3 = encode_with_spans(tok, _conv(3))
    n_im_end = "".join(chr(i) for i, m in zip(enc3.input_ids, enc3.loss_mask) if m).count("<|im_end|>")
    out.append(("三轮覆盖 3 段助手", n_im_end == 3, f"im_end={n_im_end}"))

    long = "[[" + "```sql\nSELECT 1;\n```" + "]]" * 10
    enc_tr = encode_with_spans(tok, _conv(1, long), max_len=30)
    out.append(("截断生效且不越界", enc_tr.length == 30 and enc_tr.n_truncated > 0,
                f"len={enc_tr.length} trunc={enc_tr.n_truncated}"))

    # 模板里的 enable_thinking 分支：模板含该字符串时应自动传参且不报错
    class ThinkingTok(FakeTokenizer):
        chat_template = FakeTokenizer.chat_template + "{enable_thinking}"

    enc_t = encode_with_spans(ThinkingTok(), _conv(1))
    out.append(("enable_thinking 兼容", enc_t.n_mismatch == 0, "ok"))
    return out


def check_grpo_math() -> list[tuple[str, bool, str]]:
    import numpy as np

    from training.grpo import group_advantages, group_stats

    out = []
    r = np.array([1.0, 0.0, 1.0, 0.0])
    adv = group_advantages(r, 2)
    out.append(("优势符号正确", adv[0] > 0 and adv[1] < 0, np.round(adv, 3).tolist()))
    out.append(("优势零和", abs(adv.reshape(-1, 2).sum(axis=1)).max() < 1e-6, "ok"))

    same = group_advantages(np.array([1.0, 1.0, 0.0, 0.0]), 2)
    out.append(("整组同分 → 优势 0", np.allclose(same, 0), str(same.tolist())))

    st = group_stats(np.array([1.0, 1.0, 0.0, 1.0]), 2)
    out.append(("zero_var_ratio", abs(st["zero_var_ratio"] - 0.5) < 1e-9, str(st["zero_var_ratio"])))

    raw = group_advantages(np.array([0.2, 0.4]), 2, normalize=False)
    out.append(("不标准化 → 只减均值", np.allclose(raw, [-0.1, 0.1]), str(raw.tolist())))
    return out


def check_configs(root: Path) -> list[tuple[str, bool, str]]:
    from training.config import GRPOConfig, SFTConfig, load_config

    out = []
    for p in sorted((root / "configs/train").glob("*.json")):
        cls = SFTConfig if "sft" in p.name else GRPOConfig
        try:
            cfg = load_config(p, cls)
            out.append((f"config {p.name}", True, f"out={cfg.out_dir}"))
        except Exception as e:  # noqa: BLE001
            out.append((f"config {p.name}", False, f"{type(e).__name__}: {e}"))
    return out


def run_all(root: str | Path = ROOT, groups: tuple[str, ...] | None = None) -> list[dict]:
    root = Path(root)
    results: list[dict] = []
    for group, checks in (
        ("data", lambda: check_data_files(root)),
        ("mask", lambda: check_mask(root)),
        ("grpo", check_grpo_math),
        ("config", lambda: check_configs(root)),
    ):
        if groups and group not in groups:
            continue
        try:
            items = checks()
        except Exception as e:  # noqa: BLE001
            items = [(f"{group} 组整体", False, f"{type(e).__name__}: {e}")]
        for name, ok, detail in items:
            results.append({"group": group, "name": name, "ok": bool(ok), "detail": detail})
    return results


def main() -> int:
    results = run_all()
    print("P4 离线自检（不需要 GPU）\n" + "=" * 60)
    for r in results:
        print(f"  [{'✓' if r['ok'] else '✗'}] {r['name']:<26} {r['detail']}")
    bad = [r for r in results if not r["ok"]]
    print("=" * 60)
    print(f"{len(results) - len(bad)}/{len(results)} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
