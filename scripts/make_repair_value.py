#!/usr/bin/env python3
"""合成"取值拼错"的修复轨迹（配合 ``envs/value_check.py`` 的新反馈）。

为什么需要这一批
---------------
已有的 1133 条修复轨迹只有"列名不存在"（schema 错），而线上错题 2/3 是
**执行成功但结果错**，其中很大一类是**取值写错**（`'Saless'`、`'Alamedaa'`）。
这类错误以前模型看不出来（反馈只有 "Execution succeeded"），现在值检查会明确告诉它：

    Value check: a literal in your query does not exist in the database.
    - `Saless` is not found in column `dept_name`. Closest existing values: 'Sales', 'Sales Dept'

所以可以合成"取值错 → 反馈指出 → 改对"的轨迹，补上语义错误这一类。

做法（不花钱、目标 100% 正确）
----------------------------
1. 从 gold SQL 里找一个**确实存在**的字符串字面量（用 value_check 验证）；
2. 把它改坏（删一个字符/拼写微调），执行确认结果与 gold 不同；
3. 用真实反馈（含值检查提示）+ gold 作为最后一条 assistant。

用法::

    python scripts/make_repair_value.py --limit 1200
    python scripts/make_repair_value.py --limit 20 --out data/bird/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from envs.feedback import FeedbackFormatter  # noqa: E402
from envs.sql_sandbox import SqlSandbox  # noqa: E402
from envs.value_check import check_literals, extract_literals, format_hints  # noqa: E402
from rollout.agent import Episode, Turn  # noqa: E402
from rollout.prompt import build_feedback_message  # noqa: E402

_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.S | re.I)


def extract_gold(rec: dict) -> str:
    m = _SQL_BLOCK.search(rec["messages"][-1]["content"])
    return (m.group(1).strip() if m else rec["messages"][-1]["content"].strip())


def corrupt_value(v: str, rng: random.Random) -> str | None:
    """把一个真实取值改坏：优先做"少一个字符/多一个字符/相邻交换"。"""
    if len(v) < 3:
        return None
    mode = rng.choice(["drop", "dup", "swap", "case"])
    if mode == "drop":
        i = rng.randrange(1, len(v) - 1)
        return v[:i] + v[i + 1:]
    if mode == "dup":
        i = rng.randrange(0, len(v))
        return v[:i] + v[i] + v[i:]
    if mode == "swap":
        i = rng.randrange(0, len(v) - 1)
        return v[:i] + v[i + 1] + v[i] + v[i + 2:]
    return v.upper() if not v.isupper() else v.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/bird/sft_train.jsonl")
    ap.add_argument("--db-root", default="data/raw/bird/train/train_databases")
    ap.add_argument("--out", default="data/bird/repairs_value.jsonl")
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exclude", default="data/bird/repairs_synth.jsonl",
                    help="已用于 schema 修复的题，避免一题两种修复")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (ROOT / args.sft).read_text(encoding="utf-8").splitlines() if line.strip()]
    excl = set()
    p = ROOT / args.exclude
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)["meta"]
                excl.add((m["db_id"], m["question"]))
        rows = [r for r in rows
                if (r["meta"]["db_id"], r["meta"]["question"]) not in excl]
        print(f"[排除] 已用于 schema 修复 {len(excl)} 题，剩余 {len(rows)}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]

    registry = DatabaseRegistry(ROOT / args.db_root)
    fmt = FeedbackFormatter()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    modes: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in rows:
            db = rec["meta"]["db_id"]
            gold = extract_gold(rec)
            try:
                schema = registry.schema(db)
                sb = SqlSandbox(registry.db_path(db), timeout=args.timeout,
                                max_rows=20000)
                gres = sb.run(gold)
                if not gres.success:
                    continue
                lits = extract_literals(gold)
                if not lits:
                    continue
                checks = check_literals(gold, schema, timeout=args.timeout)
                good = [c for c in checks if c.exists and not c.is_like]
                if not good:
                    continue
                ck = rng.choice(good)
                bad_val = corrupt_value(ck.literal, rng)
                if not bad_val or bad_val == ck.literal:
                    continue
                # 只替换该列的那一处字面量
                bad_sql = gold.replace(f"'{ck.literal}'", f"'{bad_val}'", 1)
                if bad_sql == gold:
                    continue
                bres = sb.run(bad_sql)
                if not bres.success:
                    continue                      # 变成报错就不算"语义错"
                if bres.rows == gres.rows:
                    continue                      # 结果没变，说明这个值不影响结果
                hints = format_hints(
                    check_literals(bad_sql, schema, timeout=args.timeout))
                fb = fmt.render_success(bres) + ("\n" + hints if hints else "")
                msgs = [rec["messages"][0], rec["messages"][1],
                        {"role": "assistant", "content": f"```sql\n{bad_sql}\n```"},
                        build_feedback_message(bad_sql, fb),
                        {"role": "assistant", "content": f"```sql\n{gold}\n```"}]
                fout.write(json.dumps({
                    "messages": msgs,
                    "meta": {
                        "db_id": db, "difficulty": rec["meta"].get("difficulty", "unknown"),
                        "n_turns": 2, "split": "train",
                        "question": rec["meta"]["question"],
                        "question_id": rec["meta"].get("question_id"),
                        "source": "bird:synthetic_value_repair",
                        "column": ck.column, "good_value": ck.literal,
                        "bad_value": bad_val, "hint_emitted": bool(hints),
                        "prompt_chars": sum(len(m["content"]) for m in msgs),
                    },
                }, ensure_ascii=False) + "\n")
                n_ok += 1
                modes["hint" if hints else "no_hint"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"[跳过] {db}: {type(e).__name__}: {e}")
                continue
            if n_ok and n_ok % 200 == 0:
                print(f"  已生成 {n_ok} 条 (含提示 {modes['hint']})", flush=True)

    print(f"[输出] {n_ok} 条 → {out_path.relative_to(ROOT)}")
    print(f"[统计] 值检查是否发出提示 {dict(modes)}")
    (out_path.parent / "repairs_value.meta.json").write_text(json.dumps({
        "n": n_ok, "source": args.sft, "limit": args.limit,
        "hint_emitted": modes.get("hint", 0),
        "error_type": "literal not found（值检查给出最近邻）",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
