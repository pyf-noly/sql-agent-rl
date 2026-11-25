#!/usr/bin/env python3
"""把"单轮 gold"与"多轮修复"合成一份 P6 用的 SFT 数据。

为什么要合而不是只训多轮
------------------------
多轮修复样本只有几百条，单轮 gold 有 3920 条；只用多轮会丢掉覆盖率，
只用单轮又学不会修复。所以按 P3/P4 的老办法混训，并用
``repeat_multiturn`` 给多轮样本加权（否则几百条会被 3920 条淹掉）。

产物（默认 ``data/bird_p6/``）
-----------------------------
- ``sft_train.jsonl``：单轮 gold + 多轮修复（多轮样本带 ``meta.n_turns``）
- ``sft_val.jsonl``：沿用 ``data/bird/sft_val.jsonl``（单轮，够用）
- ``meta.json``：构成统计 + 冲突检查（同一题不许既有单轮又有多轮）

用法::

    python scripts/merge_p6_sft.py
    python scripts/merge_p6_sft.py --repairs data/bird/repairs.jsonl --out-dir data/bird_p6
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(r: dict) -> dict:
    """两种输入格式都收：``merge`` 自己的扁平格式，以及 ``make_repair_synthetic``
    产出的 ``{messages, meta}`` 格式。"""
    if "meta" in r and "db_id" not in r:
        m = r["meta"]
        return {"db_id": m["db_id"], "question": m["question"], "messages": r["messages"],
                "difficulty": m.get("difficulty"), "question_id": m.get("question_id"),
                "source": m.get("source"),
                "mask_first_turn": bool(m.get("mask_first_turn"))}
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default="data/bird/sft_train.jsonl")
    ap.add_argument("--repairs", default="data/bird/repairs.jsonl")
    ap.add_argument("--repairs-synth", default="data/bird/repairs_synth.jsonl")
    ap.add_argument("--repairs-value", default="data/bird/repairs_value.jsonl")
    ap.add_argument("--val", default="data/bird/sft_val.jsonl")
    ap.add_argument("--out-dir", default="data/bird_p6")
    args = ap.parse_args()

    single = read_jsonl(ROOT / args.single)
    repairs = []
    for p in (args.repairs, args.repairs_synth, args.repairs_value):
        fp = ROOT / p
        if fp.is_file():
            rows = read_jsonl(fp)
            print(f"[输入] {p}: {len(rows)} 条")
            repairs.extend(rows)
    print(f"[输入] 单轮 {len(single)} 条，修复轨迹合计 {len(repairs)} 条")

    # 冲突处理：合成修复的题目本来就来自单轮数据（同一题、同一 gold）。
    # 所以不是丢掉修复，而是**用多轮版本替换掉那条单轮样本**——
    # 这样同一题只出现一次，且多轮版本里已经包含 gold（最后一条 assistant）。
    single_keys = {(r["meta"]["db_id"], r["meta"]["question"]) for r in single}
    repair_keys = set()
    multi_raw = []
    for r in repairs:
        r = normalize(r)
        k = (r["db_id"], r["question"])
        if k in repair_keys:          # 同一题只留一条修复轨迹
            continue
        repair_keys.add(k)
        multi_raw.append(r)
    replaced = single_keys & repair_keys
    single = [r for r in single
              if (r["meta"]["db_id"], r["meta"]["question"]) not in repair_keys]
    print(f"[替换] {len(replaced)} 条单轮样本被同题的多轮修复版本替换"
          f"（单轮剩 {len(single)}）")

    multi = []
    for r in multi_raw:
        r = normalize(r)
        multi.append({
            "messages": r["messages"],
            "meta": {
                "db_id": r["db_id"],
                "difficulty": r.get("difficulty") or "unknown",
                "n_turns": 2,
                "split": "train",
                "question": r["question"],
                "question_id": r.get("question_id"),
                "source": r.get("source") or "bird:repair_synth",
                # 修复样本的第一轮是故意错的，只对最后一轮算 loss（见 training/data.py）
                "mask_first_turn": bool(r.get("mask_first_turn", True)),
                "prompt_chars": sum(len(m["content"]) for m in r["messages"]),
            },
        })

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = single + multi
    with (out_dir / "sft_train.jsonl").open("w", encoding="utf-8") as f:
        for it in merged:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    val_src = ROOT / args.val
    if val_src.is_file():
        shutil.copyfile(val_src, out_dir / "sft_val.jsonl")

    turns = Counter(it["meta"]["n_turns"] for it in merged)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_single": len(single), "n_multi": len(multi), "n_total": len(merged),
        "turns_hist": dict(sorted(turns.items())),
        "replaced_single": len(replaced),
        "note": "多轮样本建议配 repeat_multiturn>=3（否则被单轮淹没）",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[输出] {out_dir.relative_to(ROOT)}/sft_train.jsonl {len(merged)} 条"
          f"（单轮 {len(single)} / 多轮 {len(multi)}）")
    print(f"[统计] 轮数分布 {meta['turns_hist']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
