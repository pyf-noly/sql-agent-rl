#!/usr/bin/env python3
"""评测器自检：把 gold SQL 当成"模型预测"喂给评测器，结果必须几乎全对。

这一步能抓出评测器的系统性 bug（列顺序、NULL、浮点、ORDER BY 语义）。
如果 gold 对 gold 都不到 99%，说明评测器本身有问题，后面所有数字都不能用。

用法::

    python scripts/check_evaluator.py --sample data/processed/dev_sample.json
    python scripts/check_evaluator.py --full      # 跑完整 1034 条
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/processed/dev_sample.json")
    ap.add_argument("--full", action="store_true", help="用完整 dev.json")
    ap.add_argument("--db-root", default="data/raw/spider/database")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.full:
        src = ROOT / "data/raw/spider/dev.json"
    else:
        src = ROOT / args.sample
    if not src.is_file():
        print(f"[失败] 找不到 {src}")
        return 1

    rows = json.loads(src.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[数据] {src.relative_to(ROOT)}  共 {len(rows)} 条")

    registry = DatabaseRegistry(ROOT / args.db_root)
    ev = Evaluator(registry)

    fails: list[dict] = []
    ok = 0
    for i, e in enumerate(rows, 1):
        try:
            r = ev.evaluate(e["db_id"], e["query"], e["query"])
        except FileNotFoundError:
            fails.append({"i": i, **e, "reason": "db_missing"})
            continue
        if r.correct:
            ok += 1
        else:
            fails.append({
                "i": i, "db_id": e["db_id"], "query": e["query"],
                "reason": r.reason, "err": r.pred_error_class,
            })
        if i % 100 == 0:
            print(f"\r      进度 {i}/{len(rows)}，通过 {ok}", end="", flush=True)
    print()

    rate = ok / len(rows) if rows else 0.0
    print(f"\n[结果] gold-as-prediction: {ok}/{len(rows)} = {rate:.2%}")
    if fails:
        print(f"[失败] {len(fails)} 条，前 5 条：")
        for f in fails[:5]:
            print(f"  - {f['db_id']}: {f['reason']} ({f.get('err')})")
            print(f"    {f['query'][:120]}")
        out = ROOT / "results" / "evaluator_selfcheck_fails.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(fails, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  完整清单: {out.relative_to(ROOT)}")

    print()
    if rate >= 0.99:
        print("[通过] 评测器自检达标（≥99%）")
        return 0
    if rate >= 0.95:
        print("[警告] 95%~99%，有几条 gold 本身跑不通或判定异常，先看失败清单再决定")
        return 2
    print("[失败] 低于 95%，评测器或数据有系统性问题，不要继续")
    return 1


if __name__ == "__main__":
    sys.exit(main())
