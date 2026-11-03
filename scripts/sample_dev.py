#!/usr/bin/env python3
"""分层抽样 dev 集，产出固定种子的评测子集。

产物::

    data/processed/dev_sample.json
        [{db_id, question, query, difficulty, idx}, ...]
    data/processed/dev_sample.meta.json
        {seed, n, per_difficulty, source, created_at}

为什么固定种子
--------------
P2 的基线、P5 的训练评测、P6 的消融，必须跑在**同一批样本**上，
否则数字之间不可比。种子写进 meta 文件，任何人重跑都能复现。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.difficulty import classify_many  # noqa: E402


def _allocate(total: int, counts: dict[str, int], keys: list[str]) -> dict[str, int]:
    """按各档实际占比分配名额，用最大余数法保证总和精确等于 total。"""
    n_all = sum(counts.get(k, 0) for k in keys)
    if n_all == 0:
        return {k: 0 for k in keys}
    raw = {k: total * counts.get(k, 0) / n_all for k in keys}
    alloc = {k: int(raw[k]) for k in keys}
    remainder = total - sum(alloc.values())
    # 余数按小数部分从大到小补
    order = sorted(keys, key=lambda k: raw[k] - alloc[k], reverse=True)
    for k in order[:remainder]:
        alloc[k] += 1
    # 不能超过该档的实际数量
    for k in keys:
        alloc[k] = min(alloc[k], counts.get(k, 0))
    # 因封顶损失的名额，补给还有余量的档
    deficit = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: counts.get(k, 0) - alloc[k], reverse=True):
        if deficit <= 0:
            break
        can = counts.get(k, 0) - alloc[k]
        add = min(can, deficit)
        alloc[k] += add
        deficit -= add
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/raw/spider/dev.json")
    ap.add_argument("--out", default="data/processed/dev_sample.json")
    ap.add_argument("--n", type=int, default=300, help="抽样总数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev_path = (ROOT / args.dev).resolve()
    if not dev_path.is_file():
        print(f"[失败] 找不到 {dev_path}，先跑 download_spider.py --meta-only")
        return 1

    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    rows = classify_many(dev)
    for i, r in enumerate(rows):
        r["idx"] = i

    keys = ["easy", "medium", "hard", "extra"]
    counts: dict[str, int] = {k: 0 for k in keys}
    for r in rows:
        counts[r["difficulty"]] += 1
    alloc = _allocate(args.n, counts, keys)
    print(f"dev 总数 {len(rows)}")
    print("总体分布:", {k: counts[k] for k in keys})
    print("抽样名额:", alloc, "合计", sum(alloc.values()))

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for k in keys:
        pool = [r for r in rows if r["difficulty"] == k]
        rng.shuffle(pool)
        picked.extend(pool[: alloc[k]])

    rng.shuffle(picked)
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = {
        "seed": args.seed,
        "n": len(picked),
        "allocated": alloc,
        "actual": {k: sum(1 for r in picked if r["difficulty"] == k) for k in keys},
        "source": str(dev_path.relative_to(ROOT)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "difficulty 由 envs/difficulty.py 按 SQL 结构自动打标，非官方标注",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n写入 {out_path}  ({len(picked)} 条)")
    print(f"写入 {meta_path}")
    print("实际分布:", meta["actual"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
