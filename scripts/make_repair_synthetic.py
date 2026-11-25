#!/usr/bin/env python3
"""用"可控的 schema 错误"合成修复轨迹（不花钱、目标 100% 正确）。

为什么要走这条路（实测教训）
--------------------------
先试的是"让强模型看反馈改本地 4B 的真实错误"，结果成功率很低：

- 4B 的错误里约 2/3 是**执行成功但结果错**（join 错字段、聚合写错），
  反馈只有 "Execution succeeded: 1 row"，教师没有任何依据去改 → 修复率 10%；
- 只留"反馈里能看出错"的（报错/超时/空结果）后修复率也只有 29%，
  而这类样本只占 30% → 900 条尝试只能产出 ~65 条多轮样本，不够训。

所以换成**构造可检测错误**：拿 gold SQL，把一个列名换成本库另一个表的真实列名
（模型最常犯的 schema 错：用了不在 FROM 里的表的列），执行会报
``no such column: X``，而反馈里带**该库所有可用列名**——这时"看反馈改对"有据可依，
目标又是 gold，**成功率 100%**，且不限量、不花钱。

覆盖的是评测里真实存在的一类失败：dev 307 条里有 23~25 条是执行报错（7.5~8%）。

产物
----
``data/bird/repairs_synth.jsonl``：每条 = 完整 4 段 messages（带真实反馈），
可直接与单轮 gold 混训（见 ``merge_p6_sft.py``）。

用法::

    python scripts/make_repair_synthetic.py --limit 1200
    python scripts/make_repair_synthetic.py --limit 30 --out data/bird/x.jsonl
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
from envs.error_taxonomy import classify_result  # noqa: E402
from envs.feedback import FeedbackFormatter  # noqa: E402
from envs.sql_sandbox import SqlSandbox  # noqa: E402
from rollout.prompt import build_feedback_message  # noqa: E402

_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.S | re.I)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_gold(rec: dict) -> str:
    m = _SQL_BLOCK.search(rec["messages"][-1]["content"])
    return (m.group(1).strip() if m else rec["messages"][-1]["content"].strip())


def replace_once(sql: str, needle: str, repl: str) -> str | None:
    pat = rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])"
    if not re.search(pat, sql):
        return None
    return re.sub(pat, repl, sql, count=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/bird/sft_train.jsonl")
    ap.add_argument("--db-root", default="data/raw/bird/train/train_databases")
    ap.add_argument("--out", default="data/bird/repairs_synth.jsonl")
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (ROOT / args.sft).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]
    print(f"[输入] 单轮 SFT 抽样 {len(rows)} 条")

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
                                max_rows=args.max_rows)
                if not sb.run(gold).success:
                    continue
                # 找出 SQL 里出现的列，以及"不在 SQL 里的真实列"（作为 alien 目标）
                used = {c for t in schema.tables for c in t.columns
                        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", gold)}
                alien = [c for t in schema.tables for c in t.columns if c not in used]
                rng.shuffle(alien)
                made = None
                for col in sorted(used, key=len, reverse=True)[:6]:
                    cands = [(a, "alien") for a in alien[:8]]
                    cands.append((col + "_x", "suffix"))
                    for repl, mode in cands:
                        bad_sql = replace_once(gold, col, repl)
                        if not bad_sql:
                            continue
                        res = sb.run(bad_sql)
                        if res.success:
                            continue
                        if "no such column" not in (res.error_message or "").lower():
                            continue
                        cls = classify_result(res.success, res.error_class,
                                              res.error_message, res.is_empty)
                        fb = fmt.render(res, cls, table_names=schema.table_names,
                                        columns=schema.columns_by_table)
                        made = (bad_sql, fb, col, repl, mode)
                        break
                    if made:
                        break
                if not made:
                    continue
                bad_sql, fb, col, repl, mode = made
                # 前两段直接复用原始单轮样本（system + user，含 evidence 与 schema），
                # 保证与推理时逐字一致；只在其后接"错误 SQL + 真实反馈 + 修复"。
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
                        "source": "bird:synthetic_repair",
                        "corrupt_column": col, "corrupt_replacement": repl, "corrupt_mode": mode,
                        "prompt_chars": sum(len(m["content"]) for m in msgs),
                    },
                }, ensure_ascii=False) + "\n")
                n_ok += 1
                modes[mode] += 1
            except Exception as e:  # noqa: BLE001
                print(f"[跳过] {db}: {type(e).__name__}: {e}")
                continue
            if n_ok and n_ok % 200 == 0:
                print(f"  已生成 {n_ok} 条", flush=True)

    print(f"[输出] {n_ok} 条 → {out_path.relative_to(ROOT)}")
    print(f"[统计] 扰动方式 {dict(modes)}")
    (out_path.parent / "repairs_synth.meta.json").write_text(json.dumps({
        "n": n_ok, "source": args.sft, "limit": args.limit,
        "modes": dict(modes), "error_type": "no such column（反馈含可用列名）",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
