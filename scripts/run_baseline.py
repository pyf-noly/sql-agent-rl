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

    # BIRD（2026-11 加入）：数据与库根目录换成 BIRD 的，prompt 用 BIRD preset
    python scripts/run_baseline.py --dataset bird --backend hf --turns 1 \\
        --timeout 10 --max-rows 20000 --tag bird_dev_t1
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
from rollout.prompt import BIRD_PROMPT  # noqa: E402

#: 数据集默认参数：数据文件、数据库根目录、prompt 配置、结果行数上限。
#: 放在一处，避免“加了 BIRD 分支却漏改某个默认值”这种静默错。
DATASETS = {
    "spider": {
        "data": "data/processed/dev_sample.json",
        "db_root": "data/raw/spider/database",
        "prompt": None,
        "max_rows": 1000,
        "compare": "rows",
    },
    "bird": {
        "data": "data/bird/dev.json",
        "db_root": "data/raw/bird/dev_20240627/dev_databases",
        "prompt": BIRD_PROMPT,
        "max_rows": 20000,
        "compare": "bird",
    },
}


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
    ds = DATASETS[args.dataset]
    p = Path(args.data_file) if args.data_file else ROOT / ds["data"]
    rows = json.loads(p.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    return rows, f"{args.dataset}_dev{len(rows)}" if args.dataset == "bird" else "dev300"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="spider")
    ap.add_argument("--data-file", default="", help="覆盖数据集默认的样本文件")
    ap.add_argument("--db-root", default="", help="覆盖数据集默认的 sqlite 根目录")
    ap.add_argument("--split", choices=["smoke", "full"], default="full")
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0,
                    help="等距抽样步长（BIRD 的 dev.json 按库排序，直接 limit 会只测前几个库）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="单条查询结果行数上限（0=用数据集默认；BIRD 默认 20000）")
    ap.add_argument("--compare", choices=["rows", "bird", ""], default="",
                    help="等价判定口径：rows=行数/归一化（Spider 默认），bird=官方集合比较")
    ap.add_argument("--value-check", action="store_true",
                    help="执行成功时检查 WHERE 字面量是否存在（只影响多轮反馈，不影响判定）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="结果文件后缀，便于区分实验")
    ap.add_argument("--until-correct", action="store_true",
                    help="多轮上限模式：判对才停，用于测自修复能力（B2）")
    ap.add_argument("--backend", choices=["api", "hf"], default="api",
                    help="api=OpenAI 兼容接口（DeepSeek/vLLM）；hf=本地 HF 模型（P4 评测 4B 用）")
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="--backend hf 时的模型名/路径")
    ap.add_argument("--adapter", default=None, help="--backend hf 时的 LoRA adapter 目录")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="--backend hf 时用 4-bit 装载（8G 显存的 4060 跑 4B 用）")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-cache", action="store_true",
                    help="禁用 LLM 磁盘缓存（测吞吐/延迟时必须开，否则第二次起全是缓存命中）")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    max_rows = args.max_rows or ds["max_rows"]
    compare = args.compare or ds["compare"]
    rows, split_name = _load_samples(args)
    if args.stride and args.stride > 1:
        # 等距抽样：BIRD 的 dev.json 按库排序，直接 --limit 会只评到前几个库。
        rows = rows[:: args.stride]
        print(f"[抽样] 等距 stride={args.stride} → {len(rows)} 条")
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

    registry = DatabaseRegistry(ROOT / (args.db_root or ds["db_root"]))
    if args.backend == "hf":
        # 本地模型：GPU 上不能多线程并发跑同一个 model，强制 workers=1。
        # 300 条单轮在 4090 上约 10~15 分钟，比租 vLLM 便宜。
        if args.workers != 1:
            print(f"[提示] --backend hf 强制 workers=1（原本 {args.workers}）")
            args.workers = 1
        from training.config import ModelConfig
        from training.model import load_for_eval
        from training.rollout import HFPolicyClient

        mc = ModelConfig(name=args.model_path, load_in_4bit=args.load_in_4bit)
        model, tok = load_for_eval(mc.name, mc, adapter=args.adapter)
        client = HFPolicyClient(model, tok, max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, name=mc.name)
        print(f"[模型] 本地 {mc.name}"
              + (" (4bit)" if args.load_in_4bit else "")
              + (f" + adapter {args.adapter}" if args.adapter else ""))
    else:
        # --no-cache 时给一个全新的临时缓存目录：进程内不会命中，跨进程也不会串。
        if args.no_cache:
            import tempfile
            cache_dir = tempfile.mkdtemp(prefix="llm-nocache-")
        else:
            cache_dir = ".cache/llm"
        client = make_client_from_env(temperature=args.temperature, max_tokens=512,
                                      cache_dir=cache_dir)
    ev = Evaluator(registry, timeout=args.timeout, max_rows=max_rows,
                   compare=compare)
    agent = SqlAgent(
        client, registry,
        max_turns=args.turns,
        timeout=args.timeout,
        temperature=args.temperature,
        stop_on_success=not args.until_correct,
        verifier=ev if args.until_correct else None,
        stop_on_correct=args.until_correct,
        prompt_cfg=ds["prompt"],
        max_rows=max_rows,
        value_check=args.value_check,
    )

    t0 = time.time()
    lock = __import__("threading").Lock()
    finished = 0

    def work(i: int) -> tuple[int, Episode, float]:
        _t = time.time()
        ep = agent.run(rows[i])
        return i, ep, time.time() - _t

    episodes: dict[int, dict] = dict(done)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                idx, ep, sec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"\n[异常] 样本 {i}: {type(e).__name__}: {e}")
                continue
            d = ep.to_dict()
            d["idx"] = idx
            d["difficulty"] = rows[idx].get("difficulty")
            d["elapsed_sec"] = round(sec, 2)
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
                    done_secs = [x.get("elapsed_sec") for x in episodes.values()
                                 if x.get("elapsed_sec")]
                    avg = (sum(done_secs) / len(done_secs)) if done_secs else 0.0
                    print(f"\r      完成 {finished}/{len(todo)}  |  EA={acc:.1%}  "
                          f"|  {avg:.1f}s/题  |  花费 ¥{client.usage.cost_cny:.3f}",
                          end="", flush=True)
                # 每 25 条落一次盘：中途崩了也能续跑，而不是全部作废。
                # 只写 episodes 这一个字段——续跑逻辑读的就是它。
                if finished % 25 == 0:
                    try:
                        out_path.write_text(
                            json.dumps({"partial": True,
                                        "episodes": [episodes[k] for k in sorted(episodes)]},
                                       ensure_ascii=False, indent=1, default=str),
                            encoding="utf-8")
                    except Exception:  # noqa: BLE001
                        pass

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
            "dataset": args.dataset,
            "split": split_name, "n": len(rows), "max_turns": args.turns,
            "stride": args.stride,
            "temperature": args.temperature, "workers": args.workers,
            "until_correct": args.until_correct,
            "timeout": args.timeout, "max_rows": max_rows,
            "compare": compare, "value_check": bool(args.value_check),
            "prompt": {
                "include_column_types": bool(ds["prompt"] and ds["prompt"].include_column_types),
                "include_fk": bool(ds["prompt"] and ds["prompt"].include_fk),
                "include_evidence": bool(ds["prompt"] and ds["prompt"].include_evidence),
                "max_schema_cols": ds["prompt"].max_schema_cols if ds["prompt"] else 60,
            },
            "model": getattr(client, "model_name", None) or str(getattr(client, "model", "unknown")),
            "base_url": getattr(client, "base_url", ""),
            "tag": tag, "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed, 1),
        },
        "usage": client.usage.to_dict(),
        "metrics": metrics.to_dict(),
        "episodes": [episodes[i] for i in sorted(episodes)],
    }
    # default=str 是保险：以后任何字段塞进非 JSON 类型，也只是写成字符串，
    # 而不是让 300 条评测结果在最后一行崩掉（这个坑刚踩过）
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")

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
