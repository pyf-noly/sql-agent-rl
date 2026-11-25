#!/usr/bin/env python3
"""用本地 4B 在 BIRD 的 SFT 库上生成"第一次尝试"，为多轮修复数据提供真实错误。

为什么要本地生成第一次尝试
------------------------
直接让 DeepSeek 做多轮修复是没用的：它一次就做对（实测 3/3 首轮正确），
根本没有"错误 → 反馈 → 改对"的轨迹可学。P3 当年靠"只抽 hard/extra"绕过去，
但 BIRD train 没有难度标注。

所以换个思路：**第一轮用我们自己 4B 的真实错误**（这才是 RL rollout 时模型会做的事），
第二轮再让强模型在同样 schema/反馈下给出正确 SQL。这样得到的轨迹：

- 第一轮的错误分布 = 线上模型的分布（不是人造错误）；
- 第二轮是"看到真实执行反馈后怎么改"，正是 P6 轮级信用分配要奖励的行为。

产物
----
``data/bird/attempts.jsonl``：每条 = 一次 4B 尝试（含 SQL、执行反馈、是否正确）
``data/bird/attempts.meta.json``：统计

不需要 API，不花钱；4060-8G 上 4-bit 约 3.7s/条。

用法::

    python scripts/gen_bird_attempts.py --limit 1500 --workers 1
    python scripts/gen_bird_attempts.py --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry, load_bird_examples  # noqa: E402
from rollout.agent import SqlAgent  # noqa: E402
from rollout.prompt import BIRD_PROMPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../.models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default="", help="可选：加 adapter（用 SFT 后的模型更贴近线上）")
    ap.add_argument("--train-json", default="data/raw/bird/train/train.json")
    ap.add_argument("--db-root", default="data/raw/bird/train/train_databases")
    ap.add_argument("--split", default="data/bird/db_split.json")
    ap.add_argument("--used-sft", default="data/bird/sft_train.jsonl",
                    help="已经用于 SFT 的题，排除掉避免重复")
    ap.add_argument("--out", default="data/bird/attempts.jsonl")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_bird_examples(ROOT / args.train_json)
    sft_dbs = set(json.loads((ROOT / args.split).read_text(encoding="utf-8"))["sft_dbs"])
    rows = [r for r in rows if r["db_id"] in sft_dbs]

    used = set()
    used_p = ROOT / args.used_sft
    if used_p.is_file():
        with open(used_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line).get("meta", {})
                    used.add((m.get("db_id"), m.get("question")))
    before = len(rows)
    rows = [r for r in rows if (r["db_id"], r["question"]) not in used]
    print(f"[过滤] SFT 库 {len(sft_dbs)} 个：{before} 条，排除已用于 SFT 的题后 {len(rows)} 条")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question_id"])
                except Exception:  # noqa: BLE001
                    continue
    todo = [r for r in rows if r.get("question_id") not in done]
    print(f"[计划] {len(rows)} 条，已完成 {len(done)}，本次 {len(todo)}")

    from training.config import ModelConfig
    from training.model import load_for_eval
    from training.rollout import HFPolicyClient

    mc = ModelConfig(name=args.model, load_in_4bit=True)
    model, tok = load_for_eval(mc.name, mc, adapter=args.adapter or None)
    client = HFPolicyClient(model, tok, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, name=mc.name)
    registry = DatabaseRegistry(ROOT / args.db_root)
    agent = SqlAgent(client, registry, max_turns=1, timeout=args.timeout,
                     temperature=args.temperature, prompt_cfg=BIRD_PROMPT,
                     max_rows=args.max_rows)
    from rollout.evaluator import Evaluator
    ev = Evaluator(registry, timeout=args.timeout, max_rows=args.max_rows, compare="bird")

    t0 = time.time()
    n_ok = 0
    turns = Counter()
    with open(out_path, "a", encoding="utf-8") as fout:
        for k, r in enumerate(todo, 1):
            ep = agent.run(r)
            t = ep.turns[0] if ep.turns else None
            correct = bool(ev.evaluate(r["db_id"], r["query"], ep.final_sql).correct) \
                if ep.final_sql else False
            n_ok += int(correct)
            rec = {
                "question_id": r.get("question_id"), "db_id": r["db_id"],
                "question": r["question"], "evidence": r.get("evidence", ""),
                "gold_sql": r["query"], "sql": ep.final_sql,
                "raw_output": (t.raw_output if t else ""),
                "success": bool(t and t.success),
                "error_class": (t.error_class if t else "parse_error"),
                "error_message": (t.error_message[:500] if t else ""),
                "row_count": (t.row_count if t else 0),
                "feedback": (t.feedback if t else ""),
                "correct": correct,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if k % 20 == 0:
                el = time.time() - t0
                print(f"\r      {k}/{len(todo)}  正确率 {n_ok/k:.0%}  "
                      f"{el/k:.1f}s/条  ETA {(len(todo)-k)*el/k/60:.0f}min", end="", flush=True)

    print()
    n_all = len(done) + len(todo)
    print(f"[完成] {len(todo)} 条，累计 {n_all}；正确率 {n_ok/max(1,len(todo)):.1%}，"
          f"耗时 {(time.time()-t0)/60:.1f} min")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "adapter": args.adapter or None,
        "temperature": args.temperature, "n": n_all,
        "wrong_attempts": n_all - n_ok,
        "acc_on_sampled": round(n_ok / max(1, len(todo)), 4),
        "seconds": round(time.time() - t0, 1),
    }
    (out_path.parent / "attempts.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
