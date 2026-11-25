#!/usr/bin/env python3
"""从已有评测结果算「服务化指标」：延迟分位数、吞吐、token、每千题成本。

为什么需要它
-----------
简历上写「单题 2.6 秒」没用，面试官会问：p95 多少？每千题多少钱？多轮之后贵多少？
这些数都能从 ``results/*.json`` 里离线算出来（每条 Episode 都存了 ``elapsed_sec``），
不需要重新跑模型。

用法::

    # 单个结果
    python scripts/latency_report.py results/baseline_bird_p6v3_t1_full.json --gpu-price 2.5

    # 多个结果对比（单轮 / 三轮 / Spider）
    python scripts/latency_report.py results/baseline_bird_p6v3_t1_full.json \
        results/baseline_bird_p6v3_t3_s5.json --gpu-price 2.5 --markdown

口径说明
--------
- 延迟：一条 Episode 的墙钟时间（含 1~3 轮生成 + 沙箱执行），不是单次生成；
- 成本（本地）：``墙钟秒数 / 3600 × GPU 单价``——即「占用这张卡多久」的机会成本；
- 成本（API）：结果文件里的 ``usage.cost_cny``；
- 每千题成本：成本 / 题数 × 1000，便于和「调用大模型 API」对比。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pct(values: list[float], p: float) -> float:
    """最近秩百分位（nearest-rank），对 300 条的小样本也稳定。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return s[k]


def summarize(path: Path, gpu_price: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    eps = data.get("episodes", [])
    cfg = data.get("config", {})
    usage = data.get("usage", {})
    metrics = data.get("metrics", {})

    n = len(eps)
    lat = [float(e.get("elapsed_sec") or 0.0) for e in eps]
    lat = [x for x in lat if x > 0]
    turns = [int(e.get("n_turns") or 1) for e in eps]
    correct = int(metrics.get("correct") or sum(1 for e in eps if e.get("correct")))
    pred_err = int(metrics.get("pred_error") or 0)

    wall = float(cfg.get("elapsed_sec") or sum(lat))
    wall_h = wall / 3600.0

    api_cost = float(usage.get("cost_cny") or 0.0)
    gpu_cost = wall_h * gpu_price
    # 本地跑用 GPU 成本，API 跑用账单成本；两者都报，让读者自己选口径
    cost = gpu_cost if gpu_cost > 0 else api_cost

    prompt_tok = int(usage.get("prompt_tokens") or 0)
    completion_tok = int(usage.get("completion_tokens") or 0)
    calls = int(usage.get("calls") or 0)

    return {
        "file": path.name,
        "n": n,
        "turns_mean": round(sum(turns) / n, 2) if n else 0.0,
        "wall_min": round(wall / 60.0, 1),
        "throughput_qpm": round(n / (wall / 60.0), 1) if wall else 0.0,
        "lat_mean": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "lat_p50": round(pct(lat, 50), 2),
        "lat_p90": round(pct(lat, 90), 2),
        "lat_p95": round(pct(lat, 95), 2),
        "lat_p99": round(pct(lat, 99), 2),
        "lat_max": round(max(lat), 2) if lat else 0.0,
        "correct": correct,
        "acc": round(correct / n, 4) if n else 0.0,
        "exec_fail": pred_err,
        "prompt_tok_per_q": round(prompt_tok / n, 1) if n and prompt_tok else 0.0,
        "completion_tok_per_q": round(completion_tok / n, 1) if n and completion_tok else 0.0,
        "calls_per_q": round(calls / n, 2) if n and calls else 0.0,
        "gpu_cost": round(gpu_cost, 3),
        "api_cost": round(api_cost, 4),
        "cost_per_q": round(cost / n, 5) if n else 0.0,
        "cost_per_1k": round(cost / n * 1000, 2) if n else 0.0,
        "cost_per_correct": round(cost / correct, 4) if correct else 0.0,
        "gpu_price": gpu_price,
    }


def print_one(r: dict) -> None:
    print(f"\n=== {r['file']} ===")
    print(f"  样本            {r['n']} 条（平均 {r['turns_mean']} 轮/题）")
    print(f"  正确率          {r['acc']*100:.2f}%（{r['correct']}/{r['n']}）"
          f"  执行失败 {r['exec_fail']}")
    print(f"  墙钟 / 吞吐     {r['wall_min']} min，{r['throughput_qpm']} 题/分钟")
    print(f"  单题延迟        mean {r['lat_mean']:.2f}s | p50 {r['lat_p50']:.2f}s | "
          f"p90 {r['lat_p90']:.2f}s | p95 {r['lat_p95']:.2f}s | "
          f"p99 {r['lat_p99']:.2f}s | max {r['lat_max']:.2f}s")
    if r["prompt_tok_per_q"]:
        print(f"  token/题        prompt {r['prompt_tok_per_q']:.0f} + "
              f"completion {r['completion_tok_per_q']:.0f}（{r['calls_per_q']} 次调用，估算）")
    print(f"  成本            GPU ¥{r['gpu_cost']:.2f}（@{r['gpu_price']}/h）"
          f" | API ¥{r['api_cost']:.4f}")
    print(f"  每千题成本      ¥{r['cost_per_1k']:.2f}")
    if r["correct"]:
        print(f"  每个正确回答    ¥{r['cost_per_correct']:.4f}")


def print_markdown(rows: list[dict]) -> None:
    print("\n| 结果 | n | 轮数 | p50 | p95 | p99 | 吞吐(题/分) | 每千题成本 | 正确率 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['file']} | {r['n']} | {r['turns_mean']} | {r['lat_p50']:.2f}s | "
              f"{r['lat_p95']:.2f}s | {r['lat_p99']:.2f}s | {r['throughput_qpm']} | "
              f"¥{r['cost_per_1k']:.2f} | {r['acc']*100:.1f}% |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="results/*.json")
    ap.add_argument("--gpu-price", type=float, default=2.5,
                    help="GPU 单价（元/卡时），默认 2.5（4090 行情）")
    ap.add_argument("--markdown", action="store_true", help="额外输出一张 markdown 对比表")
    ap.add_argument("--json", default="", help="把汇总写成 JSON 文件")
    args = ap.parse_args()

    rows = []
    for f in args.files:
        p = Path(f)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print(f"[跳过] 找不到 {f}", file=sys.stderr)
            continue
        r = summarize(p, args.gpu_price)
        rows.append(r)
        print_one(r)

    if args.markdown and len(rows) >= 2:
        print_markdown(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"\n[写出] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
