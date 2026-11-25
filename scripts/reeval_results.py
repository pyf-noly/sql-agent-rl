#!/usr/bin/env python3
"""用不同的等价判定口径，对**已有**结果文件重新算分（不需要 GPU、不重新生成）。

为什么需要它
------------
同一个模型输出，换一个判定口径分数会不一样：

- ``rows``：Spider 口径（行数敏感 + 数值归一化），P2~P5 的历史结果都是这个；
- ``bird``：BIRD 官方口径 ``set(pred) == set(gold)``（原始元组、忽略行序与重复行）。

跑 BIRD 时两条都要看：和论文比用 ``bird``，和自己的历史表比用 ``rows``。
结果文件里已经存了 ``final_sql`` 与 ``gold_sql``，所以重算只花执行时间（几百条几十秒）。

用法::

    python scripts/reeval_results.py results/baseline_bird_base4b_s3.json
    python scripts/reeval_results.py results/*.json --compare bird
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.evaluator import Evaluator, Metrics  # noqa: E402

#: 结果文件 → 用哪个 db_root 重算。命中前缀即可。
DB_ROOTS = {
    "baseline_bird": "data/raw/bird/dev_20240627/dev_databases",
    "baseline_": "data/raw/spider/database",
}


def pick_db_root(path: Path) -> str:
    for prefix, root in DB_ROOTS.items():
        if path.name.startswith(prefix):
            return root
    return "data/raw/spider/database"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", choices=["rows", "bird", "both"], default="both")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=20000)
    args = ap.parse_args()

    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"[跳过] {p} 不存在")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        eps = data.get("episodes", [])
        reg = DatabaseRegistry(ROOT / pick_db_root(p))
        modes = ["rows", "bird"] if args.compare == "both" else [args.compare]
        print(f"\n{p.name}  ({len(eps)} 条，db_root={pick_db_root(p)})")
        for mode in modes:
            ev = Evaluator(reg, timeout=args.timeout, max_rows=args.max_rows, compare=mode)
            m = Metrics()
            for e in eps:
                if not e.get("final_sql"):
                    m.add(_mk(e, False, "no_sql"), difficulty=e.get("difficulty"))
                    continue
                r = ev.evaluate(e["db_id"], e["gold_sql"], e["final_sql"])
                m.add(r, difficulty=e.get("difficulty"))
            d = m.to_dict()
            diff = {k: f"{v['accuracy']:.0%}" for k, v in d["by_difficulty"].items()}
            print(f"  {mode:5s} EA={d['accuracy']*100:5.1f}%  ({d['correct']}/{d['total']})  "
                  f"执行失败={d['pred_error']}  分难度={diff}")
    return 0


def _mk(e: dict, correct: bool, reason: str):
    from rollout.evaluator import EvalResult
    return EvalResult(correct=correct, reason=reason,
                      pred_error_class=e.get("pred_error_class"))


if __name__ == "__main__":
    sys.exit(main())
