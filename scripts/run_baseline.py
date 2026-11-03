#!/usr/bin/env python3
"""批量跑基线：并发 + 断点续跑 + 实时指标 + 成本统计。

用法::

    # 60 条摸底（单轮）
    python scripts/run_baseline.py --split smoke --turns 1

    # 300 条全量（单轮 / 三轮）
    python scripts/run_baseline.py --turns 1
    python scripts/run_baseline.py --turns 3

    # 断点续跑：同样的命令再跑一次，已完成的样本会跳过
    python scripts/run_baseline.py --turns 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.agent import Episode, SqlAgent  # noqa: E402
from rollout.evaluator import Evaluator, Metrics  # noqa: E402
from rollout.llm_client import make_client_from_env  # noqa: E402


def _load_samples(args) -> tuple[list[dict], str]:
    if args.split == "smoke":
        p = ROOT / "data/processed/dev_smoke60.json"
        if not p.is_file():
            # 从 300 里均匀取 60
            full = json.loads((ROOT / "data/processed/dev_sample.json").read_text(encoding="utf-8"))
            step = max(1, len(full) // 60)
            sub = full[::step][:60]
            p.write_text(json.dumps(sub, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[生成] {p.relative_to(ROOT)} ({len(sub)} 条)")
        return json.loads(p.read_text(encoding="utf-8")), "smoke60"
    p = ROOT / "data/processed/dev_sample.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    return rows, "dev300"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["smoke", "full"], default="full")
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="结果文件后缀，便于区分实验")
    ap.add_argument("--until-correct", action="store_true",
                    help="多轮上限模式：判对才停，用于测自修复能力（B2）")
    args = ap.parse_args()

    rows, split_name = _load_samples(args)
    tag = args.tag or f"{split_name}_t{args.turns}"
    out_path = ROOT / "results" / f"baseline_{tag}.json"
    out_path.parent.mkdir(exist_ok=True)

    # 断点续跑：读已有结果
    done: dict[int, dict] = {}
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for r in prev.get("episodes", []):
                done[r["idx"]] = r
            print(f"[续跑] 已有 {len(done)} 条结果")
        except Exception:  # noqa: BLE001
            pass

    todo = [i for i in range(len(rows)) if i not in done]
    print(f"[计划] {len(rows)} 条，待跑 {len(todo)} 条，并发 {args.workers}，"
          f"max_turns={args.turns}")

    registry = DatabaseRegistry(ROOT / "data/raw/spider/database")
    client = make_client_from_env(temperature=args.temperature, max_tokens=512)
    ev = Evaluator(registry)
    agent = SqlAgent(
        client, registry,
        max_turns=args.turns,
        timeout=args.timeout,
        temperature=args.temperature,
        stop_on_success=not args.until_correct,
        verifier=ev if args.until_correct else None,
        stop_on_correct=args.until_correct,
    )

    t0 = time.time()
    lock = __import__("threading").Lock()
    finished = 0

    def work(i: int) -> tuple[int, Episode]:
        ep = agent.run(rows[i])
        return i, ep

    episodes: dict[int, dict] = dict(done)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                idx, ep = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"\n[异常] 样本 {i}: {type(e).__name__}: {e}")
                continue
            d = ep.to_dict()
            d["idx"] = idx
            d["difficulty"] = rows[idx].get("difficulty")
            # 立刻评测
            if ep.final_sql:
                r = ev.evaluate(rows[idx]["db_id"], rows[idx]["query"], ep.final_sql)
                d["correct"] = r.correct
                d["reason"] = r.reason
                d["pred_error_class"] = r.pred_error_class
            else:
                d["correct"] = False
                d["reason"] = ep.stopped_reason
                d["pred_error_class"] = None
            with lock:
                episodes[i] = d
                finished += 1
                if finished % 10 == 0 or finished == len(todo):
                    acc = sum(1 for x in episodes.values() if x.get("correct")) / len(episodes)
                    print(f"\r      完成 {finished}/{len(todo)}  |  EA={acc:.1%}  "
                          f"|  花费 ¥{client.usage.cost_cny:.3f}", end="", flush=True)

    print()
    # 汇总
    metrics = Metrics()
    for i in sorted(episodes):
        d = episodes[i]
        metrics.add(
            _mk_result(d),
            difficulty=d.get("difficulty"),
            turns=d.get("n_turns") or 1,
        )

    elapsed = time.time() - t0
    payload = {
        "config": {
            "split": split_name, "n": len(rows), "max_turns": args.turns,
            "temperature": args.temperature, "workers": args.workers,
            "until_correct": args.until_correct,
            "model": client.model, "base_url": client.base_url,
            "tag": tag, "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed, 1),
        },
        "usage": client.usage.to_dict(),
        "metrics": metrics.to_dict(),
        "episodes": [episodes[i] for i in sorted(episodes)],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'='*56}")
    print(f"EA           {metrics.accuracy:.2%}  ({metrics.correct}/{metrics.total})")
    print(f"执行失败      {metrics.pred_error}")
    print(f"结果不符      {metrics.value_mismatch + metrics.row_mismatch}")
    diff_str = {k: f"{v['accuracy']:.0%}" for k, v in metrics.to_dict()["by_difficulty"].items()}
    print(f"分难度        {diff_str}")
    print(f"token         {client.usage.prompt_tokens} in / {client.usage.completion_tokens} out")
    print(f"花费          ¥{client.usage.cost_cny:.3f}")
    print(f"耗时          {elapsed:.0f}s")
    print(f"结果文件      {out_path.relative_to(ROOT)}")
    print(f"{'='*56}")
    return 0


def _mk_result(d: dict):
    from rollout.evaluator import EvalResult
    return EvalResult(
        correct=bool(d.get("correct")),
        reason=d.get("reason") or "unknown",
        pred_error_class=d.get("pred_error_class"),
    )


if __name__ == "__main__":
    sys.exit(main())
