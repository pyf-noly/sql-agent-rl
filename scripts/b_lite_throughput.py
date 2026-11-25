#!/usr/bin/env python3
"""B-lite：vLLM 服务化评测的吞吐实验驱动。

做什么
------
在**同一批 BIRD dev 题**上，用不同的并发数（workers）跑 ``run_baseline.py --backend api``，
对比「HF 单卡串行」与「vLLM 服务并发」的吞吐 / 延迟 / 每千题成本。

不改任何训练代码：``run_baseline.py`` 本来就支持 OpenAI 兼容后端与多线程并发，
这里只负责逐个配置起子进程、收集结果、汇总成表。

用法（在服务器上，vLLM 已起好）::

    # 先冒烟：10 条，1 个 worker
    python scripts/b_lite_throughput.py --base-url http://127.0.0.1:8000/v1 \
        --model sft --workers 1 --limit 10 --tag-prefix blite_smoke

    # 正式：220 条（stride 7），并发 1/4/8/16 各跑一遍
    python scripts/b_lite_throughput.py --base-url http://127.0.0.1:8000/v1 \
        --model sft --workers 1,4,8,16 --stride 7 --turns 1 --tag-prefix blite

产出
----
- ``results/baseline_<tag>.json``：每个并发档一份（含逐条 Episode 与 usage）
- ``results/blite_summary.json``：汇总（吞吐、p50/p95、成本）
- stdout：一张 markdown 对比表，可直接贴进报告
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.latency_report import summarize  # noqa: E402  （复用同一套延迟口径）


def run_one(args, workers: int, tag: str) -> Path | None:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_baseline.py"),
        "--dataset", args.dataset,
        "--backend", "api",
        "--turns", str(args.turns),
        "--workers", str(workers),
        "--timeout", str(args.timeout),
        "--max-rows", str(args.max_rows),
        "--stride", str(args.stride),
        "--limit", str(args.limit),
        "--tag", tag,
    ]
    if args.value_check:
        cmd.append("--value-check")
    if args.until_correct:
        cmd.append("--until-correct")
    if args.no_cache:
        # 测吞吐必须关缓存：否则第二个并发档开始全是磁盘缓存命中（p50 会假到 0.03s）
        cmd.append("--no-cache")

    env = dict(os.environ)
    env.update({
        "OPENAI_BASE_URL": args.base_url,
        "OPENAI_API_KEY": args.api_key,
        "OPENAI_MODEL": args.model,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        print(f"[失败] workers={workers} 返回码 {proc.returncode}", file=sys.stderr)
        return None
    out = ROOT / "results" / f"baseline_{tag}.json"
    return out if out.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="sft", help="vLLM 里 --served-model-name / LoRA 名")
    ap.add_argument("--workers", default="1,4,8,16", help="逗号分隔的并发数")
    ap.add_argument("--dataset", choices=["bird", "spider"], default="bird")
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--stride", type=int, default=7, help="等距抽样步长；220 条约 30~60s/档")
    ap.add_argument("--limit", type=int, default=0, help="0 = 用 stride 抽完；>0 只跑前 N 条")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--value-check", action="store_true", default=True)
    ap.add_argument("--no-value-check", dest="value_check", action="store_false")
    ap.add_argument("--until-correct", action="store_true", help="三轮才需要")
    ap.add_argument("--no-cache", action="store_true", default=True,
                    help="禁用 LLM 磁盘缓存（默认开；测吞吐必须）")
    ap.add_argument("--cache", dest="no_cache", action="store_false")
    ap.add_argument("--tag-prefix", default="blite_vllm")
    ap.add_argument("--gpu-price", type=float, default=2.5)
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = ap.parse_args()

    workers_list = [int(x) for x in args.workers.split(",") if x.strip()]
    print(f"[配置] base_url={args.base_url} model={args.model} "
          f"workers={workers_list} turns={args.turns} stride={args.stride} "
          f"value_check={args.value_check}")

    if args.dry_run:
        for w in workers_list:
            print(f"  workers={w} → tag={args.tag_prefix}_w{w}")
        return 0

    rows = []
    for w in workers_list:
        tag = f"{args.tag_prefix}_w{w}"
        out = run_one(args, w, tag)
        if out is None:
            continue
        r = summarize(out, args.gpu_price)
        r["workers"] = w
        r["config"] = "vllm"
        rows.append(r)
        print(f"[OK] workers={w}: {r['throughput_qpm']} 题/分, "
              f"p50={r['lat_p50']}s, p95={r['lat_p95']}s, EA={r['acc']*100:.1f}%")

    if rows:
        summary_path = ROOT / "results" / f"{args.tag_prefix}_summary.json"
        summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"\n[写出] {summary_path}")
        print("\n| 后端 | workers | n | p50 | p95 | 吞吐(题/分) | 每千题成本 | EA |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| vLLM | {r['workers']} | {r['n']} | {r['lat_p50']:.2f}s | "
                  f"{r['lat_p95']:.2f}s | {r['throughput_qpm']} | "
                  f"¥{r['cost_per_1k']:.2f} | {r['acc']*100:.1f}% |")
        print("\n对照：HF 单卡串行 = 23.4 题/分，p50 2.44s，p95 4.09s（全量 1534 条）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
