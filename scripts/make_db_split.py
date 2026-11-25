#!/usr/bin/env python3
"""库级切分：留出 30 个库给 RL，SFT 只在剩下的 110 个库上重训。

为什么必须这么做（P5 的核心发现）
--------------------------------
P5 实测：SFT 用过的 140 个库上，即使 hard/extra 的题，模型也常常"四条轨迹第一轮全对"。
101 步 GRPO 里只有 **26%** 的采样组有梯度，dev 子集 EA 纹丝不动（75%，与 SFT 持平）。

根因不是奖励设计，是**数据**：RL 的题来自 SFT 训过的库，模型已经会了。
奖励再设计也救不了"已经会做的题"。

解法：**库级切分**——
- SFT 只用 110 个库（模型没见过的库不能出现在 SFT 里）；
- RL 只用剩下 30 个库的题（对模型来说等于 dev 那种"新库"难度）；
- 评测仍然用 Spider dev 的 20 个库（第三组，从未参与训练）。

产出（都在 ``data/train/`` 下）
--------------------------------
- ``db_split.meta.json``：留出哪些库、各有多少题、数据量变化
- ``sft_db110/sft_train.jsonl`` / ``sft_val.jsonl``：SFT 用（不含留出库）
- ``rl_dbhold.jsonl``：RL 用（只有留出库的题）

用法::

    python scripts/make_db_split.py --holdout 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/train/sft_train.jsonl")
    ap.add_argument("--rl", default="data/train/rl_train.jsonl")
    ap.add_argument("--out-sft-dir", default="data/train/sft_db110")
    ap.add_argument("--out-rl", default="data/train/rl_dbhold.jsonl")
    ap.add_argument("--holdout", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--min-rl-per-db", type=int, default=5,
                    help="留出库至少要在这个 RL prompt 数以上，避免选的库没题可训")
    args = ap.parse_args()

    sft = _read_jsonl(ROOT / args.sft)
    rl = _read_jsonl(ROOT / args.rl)
    print(f"[读取] SFT {len(sft)} 条 / RL prompt {len(rl)} 条")

    rl_by_db = Counter(r["db_id"] for r in rl)
    sft_by_db = Counter(r["meta"]["db_id"] for r in sft)
    print(f"[分布] SFT 覆盖 {len(sft_by_db)} 个库，RL 覆盖 {len(rl_by_db)} 个库")

    # 只从"两边都有、且 RL 题量够"的库里挑，保证选出来的库真的能训
    candidates = sorted(d for d in rl_by_db
                        if rl_by_db[d] >= args.min_rl_per_db and sft_by_db.get(d, 0) > 0)
    if len(candidates) <= args.holdout:
        print(f"[失败] 可选库只有 {len(candidates)} 个，不够留出 {args.holdout} 个")
        return 1

    rng = random.Random(args.seed)
    hold = set(rng.sample(candidates, args.holdout))
    print(f"[留出] {len(hold)} 个库给 RL（候选池 {len(candidates)} 个）")

    sft_keep = [r for r in sft if r["meta"]["db_id"] not in hold]
    rl_keep = [r for r in rl if r["db_id"] in hold]

    # 重新切 train/val：原来的 val 里可能含留出库，会造成"验证集见过"
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(sft_keep)
    n_val = max(50, int(len(sft_keep) * args.val_ratio))
    val, train = sft_keep[:n_val], sft_keep[n_val:]

    out_sft = ROOT / args.out_sft_dir
    _write_jsonl(out_sft / "sft_train.jsonl", train)
    _write_jsonl(out_sft / "sft_val.jsonl", val)
    _write_jsonl(ROOT / args.out_rl, rl_keep)

    meta = {
        "seed": args.seed,
        "holdout_dbs": sorted(hold),
        "n_holdout": len(hold),
        "n_sft_train": len(train),
        "n_sft_val": len(val),
        "n_sft_removed": len(sft) - len(sft_keep),
        "n_rl_prompts": len(rl_keep),
        "rl_prompts_per_holdout_db": {d: rl_by_db[d] for d in sorted(hold)},
        "rl_difficulty": dict(Counter(r.get("difficulty") for r in rl_keep)),
        "sft_removed_ratio": round(1 - len(sft_keep) / len(sft), 3),
        "note": "SFT 与 RL 的库完全不重叠；评测用 Spider dev（第三组，从未参与训练）",
    }
    (ROOT / "data/train/db_split.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[SFT ] train {len(train)} / val {len(val)}（去掉 {meta['n_sft_removed']} 条）"
          f" -> {args.out_sft_dir}/")
    print(f"[RL  ] {len(rl_keep)} 条 -> {args.out_rl}")
    print(f"[RL  ] 难度分布 {meta['rl_difficulty']}")
    print(f"[统计] 留出库每题量 {sorted(meta['rl_prompts_per_holdout_db'].values())}")
    if len(rl_keep) < 150:
        print("[警告] 留出库的 RL prompt 不足 150 条，训练会反复见到同一批题；"
              "可加大 --holdout 或降低 --min-rl-per-db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
