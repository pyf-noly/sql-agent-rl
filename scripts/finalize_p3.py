#!/usr/bin/env python3
"""把 episodes 整合成最终交付物：SFT 训练集 + 统计 + 数据卡。

产物
----
``data/processed/sft_final.jsonl``
    最终 SFT 数据（chat 格式），已去重、打乱、切成 train/val。
    每行::

        {"messages": [...], "meta": {"db_id":..., "difficulty":..., "n_turns":..., "split": "train"}}

``data/processed/sft_stats.json``
    统计：总量、难度分布、轮数分布、db 数、token 估算、train/val 切分。

``docs/p3_data_card.md``
    数据卡：来源、生成方式、过滤规则、已知偏差。**开源时要有这个。**

用法::

    python scripts/finalize_p3.py --val-ratio 0.08 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _est_tokens(text: str) -> int:
    """粗估 token 数：英文按 4 字符/token，中文按 1.5 字符/token。

    只用于报告规模量级，不用于计费。
    """
    n = len(text)
    return max(1, n // 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/synthetic/episodes.jsonl")
    ap.add_argument("--sft", default="data/processed/sft_coldstart.jsonl")
    ap.add_argument("--out", default="data/processed/sft_final.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sft_path = ROOT / args.sft
    if not sft_path.is_file():
        print(f"[失败] 找不到 {sft_path}，先跑 synthesize_sft.py")
        return 1

    rows: list[dict] = []
    seen: set[str] = set()
    dup = 0
    for line in sft_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        msgs = d.get("messages") or []
        if len(msgs) < 2:
            continue
        # 去重键必须覆盖**整条对话**，不能只看前两轮：
        # 多轮样本常以同一个错误答案开头，只哈希首轮会把它们全杀掉。
        key = hashlib.sha256(
            "\x1e".join(f"{m['role']}\x1f{m['content']}" for m in msgs).encode("utf-8")
        ).hexdigest()
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        d["_key"] = key
        d["_chars"] = sum(len(m["content"]) for m in msgs)
        rows.append(d)

    print(f"[读取] {len(rows)} 条（去重移除 {dup} 条）")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    for i, r in enumerate(rows):
        r["meta"]["split"] = "val" if i < n_val else "train"
        r.pop("_key", None)
        r.pop("_chars", None)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 统计 ----------
    train = [r for r in rows if r["meta"]["split"] == "train"]
    val = [r for r in rows if r["meta"]["split"] == "val"]
    turns = Counter(r["meta"]["n_turns"] for r in rows)
    diff = Counter(r["meta"].get("difficulty") for r in rows)
    diff_multi = Counter(r["meta"].get("difficulty") for r in rows
                         if r["meta"]["n_turns"] >= 2)
    dbs = {r["meta"]["db_id"] for r in rows}

    total_chars = sum(sum(len(m["content"]) for m in r["messages"]) for r in rows)

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "n_unique_db": len(dbs),
        "turns": dict(sorted(turns.items())),
        "multi_turn_ratio": round(
            sum(v for k, v in turns.items() if k >= 2) / len(rows), 4) if rows else 0,
        "difficulty": dict(diff),
        "difficulty_multi_turn": dict(diff_multi),
        "est_total_tokens": _est_tokens(" " * total_chars),
        "est_avg_tokens_per_sample": _est_tokens(" " * (total_chars // max(1, len(rows)))),
        "source_episodes": str(Path(args.episodes).as_posix()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    stats_path = ROOT / "data/processed/sft_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[写入] {out_path.relative_to(ROOT)}")
    print(f"  train {len(train)} / val {len(val)}")
    print(f"  轮数分布 {stats['turns']}  多轮占比 {stats['multi_turn_ratio']:.0%}")
    print(f"  难度分布 {dict(diff)}")
    print(f"  覆盖 {len(dbs)} 个数据库")
    print(f"  估算总 token {stats['est_total_tokens']:,}，"
          f"每样本约 {stats['est_avg_tokens_per_sample']}")

    # ---------- 数据卡 ----------
    card = f"""# P3 数据卡：SFT 冷启动与 RL prompt 集

> 自动生成于 {stats['created_at']}，由 `scripts/finalize_p3.py` 产出。

## 1. 内容

| 文件 | 条数 | 用途 |
|---|---|---|
| `data/processed/sft_final.jsonl` | {len(rows)} | SFT 冷启动（chat 多轮） |
| ├ train | {len(train)} | 训练 |
| └ val | {len(val)} | 验证 |
| `data/processed/rl_prompts.json` | 1500 | RL 采样题（**无答案**） |
| `data/synthetic/episodes.jsonl` | 原始轨迹 | 审计用 |

- 覆盖数据库：{len(dbs)} 个（train split 的库，与 dev 的 20 个库不重叠）
- 难度分布：{dict(diff)}
- 轮数分布：{stats['turns']}，多轮占比 **{stats['multi_turn_ratio']:.0%}**
- 估算规模：{stats['est_total_tokens']:,} tokens

## 2. 生成方式

1. **种子**：Spider 1.0 train 集，剔除 3 条 gold SQL 本身无法执行的脏数据
   （2 条 `INTERSECT` 前误用 `ORDER BY`，1 条库中缺表）；
2. **采样**：强模型（DeepSeek-chat）在「schema + 问题」上多轮生成，温度 0.7；
3. **反馈**：每轮把 SQLite 执行结果或结构化报错回灌，**不含 gold 答案**；
4. **停止**：判对即停（最多 3 轮）；
5. **过滤**：只保留最终判对的轨迹；保留中间失败轮，让模型学到"改错"。

## 3. 已知偏差（必须写清楚）

1. **蒸馏偏差**：轨迹由强模型生成，SFT 后的 4B 模型最多逼近该模型水平；
2. **判对口径**：用单库执行等价判定，比官方 test-suite 宽松，可能混入少量
   "结果碰巧相同"的错误 SQL；
3. **难度标签是自造的**：`envs/difficulty.py` 按 SQL 结构分档，不是 Spider 官方标注；
4. **多轮样本只覆盖成功的修正**：255 条跑满 3 轮仍错的轨迹没有进 SFT
   （无成功答案），它们是 RL 阶段的主要学习对象。

## 4. 与 RL prompt 集的隔离

`rl_prompts.json` 与 `sft_final.jsonl` **来自不同种子的互斥子集**，
避免 RL 阶段在 SFT 见过的题上过拟合。两者都不含 dev 的 20 个库。
"""
    card_path = ROOT / "docs/p3_data_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    print(f"[写入] {card_path.relative_to(ROOT)}")

    # 一致性检查
    if len(rows) < 500:
        print("[警告] SFT 样本少于 500 条，4B 模型可能学不出东西")
    n_multi = sum(v for k, v in turns.items() if k >= 2)
    if stats["multi_turn_ratio"] < 0.1:
        print(f"[警告] 多轮样本只有 {n_multi} 条（{stats['multi_turn_ratio']:.1%}），"
              f"不足以教会自修复——训练时需提高多轮样本权重，或依赖 RL 阶段补足")
    return 0


if __name__ == "__main__":
    sys.exit(main())
