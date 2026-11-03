#!/usr/bin/env python3
"""准备 P3 的两类数据：RL prompt 集 + SFT 种子题。

产物
----
``data/processed/rl_prompts.json``
    1500 条训练题，**只有 db_id + question**（外加 difficulty 便于分析）。
    RL 阶段喂给模型让它自己采样，绝不包含 gold 答案。

``data/processed/sft_seeds.json``
    3000 条训练题，用于合成 SFT 轨迹。和 rl_prompts **不重叠**。

``data/processed/excluded.json``
    gold SQL 无法执行的题目（数据库缺表 / Spider 标注错误），必须剔除。

用法::

    python scripts/prep_p3_data.py --rl 1500 --sft 3000 --seed 42
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

from envs.db_registry import DatabaseRegistry  # noqa: E402
from envs.difficulty import classify_many  # noqa: E402
from envs.sql_sandbox import SqlSandbox  # noqa: E402


def _verify_gold(rows: list[dict], reg: DatabaseRegistry) -> tuple[list[dict], list[dict]]:
    """剔除 gold SQL 跑不通的题。返回 (可用, 被剔除)。"""
    good, bad = [], []
    for e in rows:
        r = SqlSandbox(reg.db_path(e["db_id"])).run(e["query"])
        if r.success:
            good.append(e)
        else:
            bad.append({**e, "exclude_reason": r.error_message[:200],
                        "exclude_class": r.error_class})
    return good, bad


def _stratified_pick(rows: list[dict], n: int, rng: random.Random,
                     difficulties: list[str] | None = None) -> list[dict]:
    """按 difficulty 比例抽 n 条。"""
    keys = difficulties or ["easy", "medium", "hard", "extra"]
    counts = {k: sum(1 for r in rows if r["difficulty"] == k) for k in keys}
    total = sum(counts.values())
    if total == 0:
        return []
    alloc = {k: int(n * counts[k] / total) for k in keys}
    for k in keys:  # 补齐余数
        if sum(alloc.values()) >= n:
            break
        if counts[k] > alloc[k]:
            alloc[k] += 1
    picked: list[dict] = []
    for k in keys:
        pool = [r for r in rows if r["difficulty"] == k]
        rng.shuffle(pool)
        picked.extend(pool[: alloc[k]])
    rng.shuffle(picked)
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/spider/train_spider.json")
    ap.add_argument("--rl", type=int, default=1500, help="RL prompt 条数")
    ap.add_argument("--sft", type=int, default=3000, help="SFT 种子题数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    train_path = ROOT / args.train
    if not train_path.is_file():
        print(f"[失败] 找不到 {train_path}")
        return 1

    registry = DatabaseRegistry(ROOT / "data/raw/spider/database")
    raw = json.loads(train_path.read_text(encoding="utf-8"))
    print(f"[读取] {train_path.relative_to(ROOT)}: {len(raw)} 条")

    labeled = classify_many(raw)
    # 给每条一个稳定的键，后续做集合差集用（classify_many 会返回新对象，不能靠 id）
    for i, r in enumerate(labeled):
        r["_key"] = i
    print("[校验] 检查 gold SQL 是否可执行 ...")
    good, bad = _verify_gold(labeled, registry)
    print(f"       可用 {len(good)}，剔除 {len(bad)}")

    rng = random.Random(args.seed)
    rng.shuffle(good)
    need = args.rl + args.sft
    if len(good) < need:
        print(f"[警告] 可用样本只有 {len(good)}，少于请求的 {need}，按比例缩减")
        args.rl = int(args.rl * len(good) / need)
        args.sft = len(good) - args.rl

    rl_rows = _stratified_pick(good, args.rl, rng)
    # 互斥键用 (db_id, question) 而不是内部序号：
    # Spider train 里存在完全相同的题目，按序号去重会漏掉重复题的另一个副本，
    # 导致同一道题同时出现在 RL 和 SFT 里（RL 评测虚高的隐患）。
    used = {(r["db_id"], r["question"]) for r in rl_rows}
    sft_pool = [r for r in good if (r["db_id"], r["question"]) not in used]
    sft_rows = _stratified_pick(sft_pool, args.sft, rng)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _strip(rows: list[dict], keep_query: bool) -> list[dict]:
        out = []
        for r in rows:
            d = {"db_id": r["db_id"], "question": r["question"],
                 "difficulty": r.get("difficulty")}
            if keep_query:
                d["query"] = r["query"]
            out.append(d)
        return out

    rl_path = out_dir / "rl_prompts.json"
    rl_path.write_text(json.dumps(_strip(rl_rows, keep_query=False),
                                  ensure_ascii=False, indent=1), encoding="utf-8")

    sft_path = out_dir / "sft_seeds.json"
    sft_path.write_text(json.dumps(_strip(sft_rows, keep_query=True),
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    ex_path = out_dir / "excluded.json"
    ex_path.write_text(json.dumps(
        [{k: e.get(k) for k in ("db_id", "question", "query",
                                "exclude_reason", "exclude_class")} for e in bad],
        ensure_ascii=False, indent=1), encoding="utf-8")

    meta = {
        "seed": args.seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(train_path.relative_to(ROOT)),
        "n_source": len(raw),
        "n_usable": len(good),
        "n_excluded": len(bad),
        "n_rl": len(rl_rows),
        "n_sft": len(sft_rows),
        "rl_diff": {k: sum(1 for r in rl_rows if r["difficulty"] == k)
                    for k in ("easy", "medium", "hard", "extra")},
        "sft_diff": {k: sum(1 for r in sft_rows if r["difficulty"] == k)
                     for k in ("easy", "medium", "hard", "extra")},
        "note": "rl_prompts 不含 gold 答案；sft_seeds 含 gold 供合成时校验",
    }
    (out_dir / "p3_data.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[写入] {rl_path.relative_to(ROOT)}  ({len(rl_rows)} 条，无答案)")
    print(f"[写入] {sft_path.relative_to(ROOT)}  ({len(sft_rows)} 条，含 gold)")
    print(f"[写入] {ex_path.relative_to(ROOT)}  ({len(bad)} 条被剔除)")
    print("\n分布:")
    print("  rl_prompts:", meta["rl_diff"])
    print("  sft_seeds :", meta["sft_diff"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
