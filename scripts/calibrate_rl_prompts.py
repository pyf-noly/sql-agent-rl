#!/usr/bin/env python3
"""P5 第一步：给 RL prompt 做 pass-rate 校准，只留下"有梯度信号"的题。

为什么必须先做这一步
-------------------
P4 冒烟测到 **zero_var_ratio = 0.90**：训练集上 reward 均值 0.95，G=4 时
"四条全对"的概率约 0.95⁴ ≈ 0.81，加上全错的组，90% 的采样对参数更新零贡献。

GRPO 的优势是组内相对值：

    A_i = (r_i - mean(r)) / std(r)

一组全对或全错 → std=0 → A 全 0 → 这条数据白采。
所以要先把 prompt 分成三类，只训中间那类：

    p = 0        → 太简单/太难，训了没用（全对或全错）
    0 < p < 1    → **有信号**，这才是 RL 的燃料
    p = 1        → 全对，没信号

按经验，保留 ``p ∈ [0.2, 0.8]`` 最好：太低（全错）和太高（全对）两边都学不到东西，
中间区间每组平均能贡献 1~3 个"反向"样本。

产出
----
- ``data/train/rl_calibrated.jsonl``：带 ``p`` 字段的全部 prompt（可复查）
- ``data/train/rl_p5_train.jsonl``：筛出来的训练集（0.2 ≤ p ≤ 0.8）
- ``data/train/rl_calibration.meta.json``：分布统计 + 筛了多少 + 预计零方差比例

断点续跑
--------
每条 prompt 跑完就追加一行；中断后重跑同一条命令会跳过已完成的 id。
校准在 4090 上大概 30~50 分钟，这个续跑能力是必须的。

用法::

    # 先只校准 hard/extra（396 条），最省时间
    python scripts/calibrate_rl_prompts.py --difficulty hard,extra --group-size 4

    # 不够用再校准全量
    python scripts/calibrate_rl_prompts.py --difficulty all --group-size 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stable_seed(text: str) -> int:
    """稳定的整数种子。

    不能用内置 ``hash()``：Python 的字符串哈希每个进程都加了随机盐，
    同一道题两次运行会采到不同的样本，校准就不可复现了。
    """
    return zlib.crc32(text.encode("utf-8")) % 100_000


def _load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    done: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except Exception:  # noqa: BLE001
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-file", default="data/train/rl_train.jsonl")
    ap.add_argument("--out", default="data/train/rl_calibrated.jsonl")
    ap.add_argument("--train-out", default="data/train/rl_p5_train.jsonl")
    ap.add_argument("--difficulty", default="hard,extra",
                    help="逗号分隔的难度；all 表示全部")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--p-low", type=float, default=0.2)
    ap.add_argument("--p-high", type=float, default=0.8)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default="runs/sft_qwen3_4b/adapter")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch

    from envs.db_registry import DatabaseRegistry
    from training.config import ModelConfig
    from training.model import load_for_eval
    from training.rollout import HFPolicyClient, episode_reward, make_agent

    rl_path = Path(args.rl_file)
    if not rl_path.is_absolute():
        rl_path = ROOT / rl_path
    rows = [json.loads(line) for line in rl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.difficulty != "all":
        keep = {x.strip() for x in args.difficulty.split(",") if x.strip()}
        rows = [r for r in rows if r.get("difficulty") in keep]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[数据] {rl_path.name} 里符合条件的 prompt {len(rows)} 条"
          f"（难度 {args.difficulty}）")

    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[续跑] 已完成 {len(done)} 条，待跑 {len(todo)} 条")

    if not todo:
        print("[跳过] 校准已全部完成，直接生成筛选集")
        _finalize(done, rows, args)
        return 0

    print(f"[模型] {args.model} + {args.adapter}")
    mc = ModelConfig(name=args.model)
    model, tok = load_for_eval(mc.name, mc, adapter=args.adapter)
    registry = DatabaseRegistry(ROOT / "data/raw/spider/database")
    client = HFPolicyClient(model, tok, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, name=args.model)
    agent, ev = make_agent(client, registry, max_turns=args.max_turns,
                           timeout=2.0, temperature=args.temperature)

    t0 = time.time()
    # 续跑时文件末尾可能是上次崩溃留下的半行；先补一个换行，避免把新记录接在半行后面
    if out_path.is_file() and out_path.stat().st_size > 0:
        with out_path.open("rb") as fb:
            fb.seek(-1, 2)
            if fb.read(1) != b"\n":
                with out_path.open("a", encoding="utf-8") as fa:
                    fa.write("\n")

    with out_path.open("a", encoding="utf-8") as f:
        for i, sample in enumerate(todo):
            rewards, turns, first_ok = [], [], 0
            for g in range(args.group_size):
                torch.manual_seed(args.seed * 1_000_003 + _stable_seed(sample["id"]) * 10 + g)
                ep = agent.run(sample)
                r, _ = episode_reward(ep, sample, ev)
                rewards.append(r)
                turns.append(ep.n_turns)
                # 首轮是否就答对：这是"题目难度"更干净的刻度。
                # p3（三轮后能否对）被 87% 的准确率抬得太高，p1 才能看出真难度。
                if ep.turns and ep.turns[0].correct:
                    first_ok += 1

            p = sum(rewards) / len(rewards)
            rec = {
                "id": sample["id"],
                "db_id": sample["db_id"],
                "question": sample["question"],
                "query": sample["query"],
                "difficulty": sample.get("difficulty"),
                "p": round(p, 4),
                "p1": round(first_ok / len(rewards), 4),
                "n": len(rewards),
                "n_correct": int(sum(rewards)),
                "turns_mean": round(statistics.mean(turns), 2),
            }
            done[sample["id"]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                p_list = [x["p"] for x in done.values()]
                keep_n = sum(1 for x in p_list if args.p_low <= x <= args.p_high)
                eta = el / (i + 1) * (len(todo) - i - 1) / 60
                print(f"\r  {i + 1}/{len(todo)}  |  当前有信号 {keep_n}/{len(p_list)} "
                      f"({keep_n / max(1, len(p_list)):.0%})  |  已用 {el / 60:.0f}min "
                      f"|  预计还需 {eta:.0f}min", end="", flush=True)

    print()
    _finalize(done, rows, args)
    return 0


def _rel(p: Path) -> str:
    """尽量给相对路径；输出目录在仓库外时退回绝对路径（别在最后一行崩掉）。"""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _finalize(done: dict[str, dict], rows: list[dict], args) -> None:
    """统计分布 + 写筛选集。"""
    recs = [done[r["id"]] for r in rows if r["id"] in done]
    if not recs:
        print("[失败] 没有任何校准结果")
        return

    p_list = [r["p"] for r in recs]
    hist = Counter(round(p, 2) for p in p_list)
    kept = [r for r in recs if args.p_low <= r["p"] <= args.p_high]
    all_correct = sum(1 for p in p_list if p >= 0.999)
    all_wrong = sum(1 for p in p_list if p <= 0.001)

    # 校准集上的"理论零方差比例"：p^G + (1-p)^G 的平均
    g = args.group_size
    zero_var = statistics.mean(p ** g + (1 - p) ** g for p in p_list)

    meta = {
        "n_calibrated": len(recs),
        "n_kept": len(kept),
        "keep_ratio": round(len(kept) / len(recs), 3),
        "p_mean": round(statistics.mean(p_list), 4),
        "p_median": round(statistics.median(p_list), 4),
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "theoretical_zero_var_ratio_at_G": round(zero_var, 3),
        "hist": {str(k): v for k, v in sorted(hist.items())},
        "difficulty": dict(Counter(r["difficulty"] for r in kept)),
        "p_range": [args.p_low, args.p_high],
        "group_size": g,
    }

    # ---- 首轮通过率 p1 的统计（老记录没有 p1 字段，自动跳过）----
    p1_recs = [r for r in recs if "p1" in r]
    kept1: list[dict] = []
    if p1_recs:
        p1_list = [r["p1"] for r in p1_recs]
        kept1 = [r for r in p1_recs if args.p_low <= r["p1"] <= args.p_high]
        meta.update({
            "p1_mean": round(statistics.mean(p1_list), 4),
            "p1_median": round(statistics.median(p1_list), 4),
            "n_kept_by_p1": len(kept1),
            "p1_all_correct": sum(1 for x in p1_list if x >= 0.999),
            "p1_zero_var_ratio_at_G": round(
                statistics.mean(x ** g + (1 - x) ** g for x in p1_list), 3),
        })

    train_path = Path(args.train_out)
    train_path = train_path if train_path.is_absolute() else ROOT / train_path
    with train_path.open("w", encoding="utf-8") as f:
        for r in kept:
            # 训练文件保留原始 prompt 字段，并带上校准得到的 p
            f.write(json.dumps({
                "id": r["id"], "db_id": r["db_id"], "question": r["question"],
                "query": r["query"], "difficulty": r["difficulty"],
                "p": r["p"], "p1": r.get("p1"), "turns_mean": r["turns_mean"],
            }, ensure_ascii=False) + "\n")

    # 另一份：按**首轮**通过率筛（p1 更能反映真实难度，见脚本注释）
    train_path_p1 = train_path.with_name(train_path.stem + "_p1.jsonl")
    if kept1:
        with train_path_p1.open("w", encoding="utf-8") as f:
            for r in kept1:
                f.write(json.dumps({
                    "id": r["id"], "db_id": r["db_id"], "question": r["question"],
                    "query": r["query"], "difficulty": r["difficulty"],
                    "p": r["p"], "p1": r["p1"], "turns_mean": r["turns_mean"],
                }, ensure_ascii=False) + "\n")

    meta_path = train_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"校准 {meta['n_calibrated']} 条  |  p 均值 {meta['p_mean']:.3f} "
          f"中位数 {meta['p_median']:.3f}")
    print(f"全对 {all_correct} 条 / 全错 {all_wrong} 条 / 有信号区间内 {len(kept)} 条 "
          f"({meta['keep_ratio']:.0%})")
    print(f"按 G={g} 估算的零方差比例（校准集上）：{meta['theoretical_zero_var_ratio_at_G']:.2f}")
    if p1_recs:
        print(f"[首轮] p1 均值 {meta['p1_mean']:.3f} 中位数 {meta['p1_median']:.3f} | "
              f"首轮全对 {meta['p1_all_correct']} 条 | "
              f"按 p1 筛出 {meta['n_kept_by_p1']} 条 | "
              f"零方差 {meta['p1_zero_var_ratio_at_G']:.2f}")
    print(f"筛选集难度分布：{meta['difficulty']}")
    print(f"训练集(p3) -> {_rel(train_path)}  ({len(kept)} 条)")
    if kept1:
        print(f"训练集(p1) -> {_rel(train_path_p1)}  ({len(kept1)} 条)")
    print(f"统计   -> {_rel(meta_path)}")
    print("=" * 60)

    best = max(len(kept), len(kept1))
    if best < 50:
        print("[警告] 有信号的题不足 50 条，outcome-only GRPO 很难训出东西。"
              "建议：① 降 --p-low 到 0.1；② 用 p1 筛（更接近真实难度）；"
              "③ 换轮级塑形奖励（configs/train/grpo_p5_dense.json，见 guides/P5.md 第 8 节）")


if __name__ == "__main__":
    sys.exit(main())
