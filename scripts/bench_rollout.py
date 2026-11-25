#!/usr/bin/env python3
"""B-full 第一步：rollout 采样基准（同步 HF vs 异步 vLLM）。

训练每步的耗时 = 一条 prompt 的 G 条轨迹 + 一次参数更新。更新很小（<2s），
所以瓶颈几乎全在 rollout。这个脚本只测 rollout 部分：

- ``hf_sync``：和 ``train_grpo`` 现在的做法一样，G 条**串行**跑；
- ``hf_thread``：G 条并发，但仍是同一个 HF 模型（对照组，看并发对 HF 有没有用）；
- ``vllm_async``：G 条并发请求 vLLM 服务。

同一批题、同样的 agent 配置（3 轮 / 判对才停 / value_check / T=1.0）。

用法::

    # 1) 同步 HF（先跑，vLLM 服务此时要关掉，否则抢显存）
    python scripts/bench_rollout.py --backend hf --mode hf_sync --prompts 25 --group 4

    # 2) 起 vLLM 后
    python scripts/bench_rollout.py --backend vllm --mode vllm_async \
        --base-url http://127.0.0.1:8000/v1 --model qwen3-4b --concurrency 4

输出：每个 episode 的延迟、每个 group（=一个训练步的 rollout）的墙钟、吞吐、轮数。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.agent import SqlAgent  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402
from scripts.run_baseline import DATASETS  # noqa: E402


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))]


def build_agent(args, registry):
    ds = DATASETS[args.dataset]
    ev = Evaluator(registry, timeout=args.timeout, max_rows=args.max_rows,
                   compare=ds["compare"])
    if args.backend == "hf":
        from training.config import ModelConfig
        from training.model import load_for_eval
        from training.rollout import HFPolicyClient

        if not args.model_path:
            raise SystemExit("--backend hf 需要 --model-path")
        mc = ModelConfig(name=args.model_path, load_in_4bit=args.load_in_4bit)
        t0 = time.perf_counter()
        model, tok = load_for_eval(mc.name, mc, adapter=args.adapter)
        print(f"[hf] 模型加载 {time.perf_counter()-t0:.1f}s", flush=True)
        client = HFPolicyClient(model, tok, max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, name=mc.name)
    else:
        from rollout.llm_client import LLMClient

        client = LLMClient(base_url=args.base_url, api_key="EMPTY", model=args.model,
                           cache_dir=tempfile.mkdtemp(prefix="bench-"),
                           temperature=args.temperature, max_tokens=args.max_new_tokens)
    agent = SqlAgent(
        client, registry, max_turns=args.turns, timeout=args.timeout,
        temperature=args.temperature, stop_on_success=False, stop_on_correct=True,
        verifier=ev, prompt_cfg=ds["prompt"], max_rows=args.max_rows,
        value_check=args.value_check,
    )
    return agent, client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="bird")
    ap.add_argument("--mode", choices=["hf_sync", "hf_thread", "vllm_async"], required=True)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--prompts", type=int, default=25, help="题数")
    ap.add_argument("--stride", type=int, default=0, help="0 = 自动按 --prompts 等距取")
    ap.add_argument("--group", type=int, default=4, help="G")
    ap.add_argument("--concurrency", type=int, default=0, help="0 = 等于 group")
    ap.add_argument("--parallel-prompts", type=int, default=1,
                    help="同时处理几条 prompt（真实异步 rollout 会预取多条；1 = 只并发 G 条）")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--value-check", action="store_true", default=True)
    ap.add_argument("--no-value-check", dest="value_check", action="store_false")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    rows = json.loads((ROOT / ds["data"]).read_text(encoding="utf-8"))
    stride = args.stride or max(1, len(rows) // args.prompts)
    rows = rows[::stride][: args.prompts]
    print(f"[数据] {len(rows)} 条（stride={stride}），G={args.group}，max_turns={args.turns}，"
          f"T={args.temperature}", flush=True)

    registry = DatabaseRegistry(ROOT / ds["db_root"])
    agent, client = build_agent(args, registry)

    conc = args.concurrency or args.group
    print(f"[模式] {args.mode}，并发 {conc if args.mode != 'hf_sync' else 1}", flush=True)

    episodes: list[dict] = []
    group_walls: list[float] = []

    def run_one(sample: dict) -> dict:
        t0 = time.perf_counter()
        ep = agent.run(sample)
        return {"turns": ep.n_turns, "lat": time.perf_counter() - t0,
                "stopped": ep.stopped_reason, "db": sample["db_id"]}

    t_all = time.perf_counter()
    K = max(1, args.parallel_prompts)
    for c0 in range(0, len(rows), K):
        chunk = rows[c0:c0 + K]
        t_group = time.perf_counter()
        if args.mode == "hf_sync":
            results = [run_one(sample) for sample in chunk for _ in range(args.group)]
        else:
            tasks = [(sample, rep) for sample in chunk for rep in range(args.group)]
            workers = min(conc * K, len(tasks)) if args.mode != "hf_thread" else conc
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                results = list(pool.map(lambda t: run_one(t[0]), tasks))
        # 这一"批"的墙钟 / K = 每条 prompt 等效的 rollout 时间
        chunk_wall = time.perf_counter() - t_group
        group_walls.extend([chunk_wall / len(chunk)] * len(chunk))
        episodes.extend(results)
        if c0 == 0 or (c0 // K + 1) % 5 == 0:
            done = len(episodes)
            print(f"  [{min(c0+K, len(rows))}/{len(rows)} prompts] "
                  f"chunk {chunk_wall:.2f}s | 已跑 {done} 条 | "
                  f"累计 {time.perf_counter()-t_all:.0f}s", flush=True)

    wall = time.perf_counter() - t_all
    lats = [e["lat"] for e in episodes]
    turns = [e["turns"] for e in episodes]
    usage = {}
    try:
        usage = client.usage.to_dict()
    except Exception:  # noqa: BLE001
        pass

    out = {
        "mode": args.mode, "dataset": args.dataset, "prompts": len(rows),
        "group": args.group, "concurrency": conc if args.mode != "hf_sync" else 1,
        "turns_max": args.turns, "wall_s": round(wall, 2),
        "episodes": len(episodes),
        "episodes_per_s": round(len(episodes) / wall, 3),
        "samples_per_s": round(len(episodes) / wall, 3),
        "group_p50": round(pct(group_walls, 50), 2),
        "group_p95": round(pct(group_walls, 95), 2),
        "group_mean": round(statistics.mean(group_walls), 2),
        "lat_p50": round(pct(lats, 50), 2), "lat_p95": round(pct(lats, 95), 2),
        "turns_mean": round(statistics.mean(turns), 2),
        "usage": usage,
    }
    print("\n=== 结果 ===")
    for k, v in out.items():
        if k != "usage":
            print(f"  {k}: {v}")
    if usage:
        print(f"  tokens: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out")

    print(f"\n折算：每步 rollout = group_p50 {out['group_p50']:.2f}s "
          f"（训练每步 = 这个数 + 更新，更新通常 <2s）")
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[写出] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
