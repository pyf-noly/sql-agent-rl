#!/usr/bin/env python3
"""BIRD 数据与评测口径的离线自检（不需要 GPU、不花钱）。

为什么单独一个脚本
------------------
Spider 那套在 P4 有 ``selfcheck_p4.py``；BIRD 换了数据源、换了 JSON 字段、换了
prompt（类型/外键/evidence/引号），任何一个环节错了都会在租来的卡上白白烧钱。
所以上卡前先在本地跑这一遍，20 秒出结论。

检查项（每项失败都会让退出码非 0）
--------------------------------
1. 样本数与字段完整性（question_id/db_id/question/query/difficulty）
2. 每条样本的 sqlite 文件存在
3. gold 可执行率（抽查 N 条，超时单列）
4. ``gold 当预测`` 必须 100% 判对（评测器自检，防止规则把等价当不等价）
5. prompt 里不许出现 gold SQL（反泄漏）
6. prompt 里的 DDL 没有被 6000 字符截断
7. evidence 存在时确实进了 prompt
8. 难度映射取值域正确（easy/medium/hard）

用法::

    python scripts/selfcheck_bird.py
    python scripts/selfcheck_bird.py --limit 200 --gold-check 50
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry, load_bird_examples  # noqa: E402
from envs.sql_sandbox import SqlSandbox  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402
from rollout.prompt import BIRD_PROMPT, build_messages  # noqa: E402

OK = "\u2713"
BAD = "\u2717"


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[bool, str]] = []

    def check(self, ok: bool, name: str) -> bool:
        self.items.append((ok, name))
        print(f"  [{OK if ok else BAD}] {name}")
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for ok, _ in self.items if not ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bird/dev.json")
    ap.add_argument("--db-root", default="data/raw/bird/dev_20240627/dev_databases")
    ap.add_argument("--raw", default="data/raw/bird/dev_20240627/dev.json",
                    help="BIRD 官方 dev.json（用来交叉校验字段映射）")
    ap.add_argument("--limit", type=int, default=0, help="只检查前 N 条（0=全量）")
    ap.add_argument("--gold-check", type=int, default=200, help="抽查多少条 gold 可执行")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rep = Report()
    data_path = ROOT / args.data
    if not data_path.is_file():
        print(f"找不到 {data_path}，先跑 scripts/prepare_bird.py")
        return 2
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    reg = DatabaseRegistry(ROOT / args.db_root)
    print(f"[数据] {data_path.relative_to(ROOT)}：{len(rows)} 条 / "
          f"{len({r['db_id'] for r in rows})} 个库")

    # 1. 字段
    need = {"question_id", "db_id", "question", "query", "difficulty", "evidence"}
    missing = [i for i, r in enumerate(rows) if need - set(r)]
    rep.check(not missing, f"字段完整（缺字段样本 {len(missing)}）")
    rep.check(all(str(r["query"]).strip().upper().startswith(("SELECT", "WITH"))
                  for r in rows), "所有 gold 都是 SELECT/WITH")
    diff = Counter(r["difficulty"] for r in rows)
    rep.check(set(diff) <= {"easy", "medium", "hard"}, f"难度取值合法 {dict(diff)}")

    # 2. 数据库文件
    absent = [d for d in {r["db_id"] for r in rows}
              if not (ROOT / args.db_root / d / f"{d}.sqlite").is_file()]
    rep.check(not absent, f"sqlite 文件齐全（缺 {len(absent)} 个：{absent[:3]}）")

    # 3. gold 可执行（随机抽查）
    # 用 30s 超时而不是评测用的 10s：这一步是验"数据本身能不能跑"，
    # 不想因为两条本来就慢的 BIRD 查询（card_games 518 / codebase_community 701）误报。
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.gold_check, len(rows)))
    fails, timeouts = [], 0
    for r in sample:
        res = SqlSandbox(reg.db_path(r["db_id"]), timeout=30.0, max_rows=20000).run(r["query"])
        if not res.success:
            if res.error_class == "timeout":
                timeouts += 1
            else:
                fails.append((r["question_id"], res.error_class, res.error_message[:60]))
    rep.check(len(fails) == 0,
              f"gold 抽查 {len(sample)} 条可执行（失败 {len(fails)}，超时 {timeouts}）")
    if fails:
        for f in fails[:3]:
            print("      ", f)

    # 4. gold-as-prediction 必须全对
    # 例外：极少数 gold 本身就跑不完（BIRD 有 1~2 条，即使 30s 也超时）。
    # 这是数据属性、不是我们的 bug：官方评测里它们同样算错，所以这里单独计数并跳过。
    ev = Evaluator(reg, timeout=30.0, max_rows=20000)
    n_bad, n_slow = 0, 0
    for r in sample:
        res = ev.evaluate(r["db_id"], r["query"], r["query"])
        if res.correct:
            continue
        g = SqlSandbox(reg.db_path(r["db_id"]), timeout=30.0, max_rows=20000).run(r["query"])
        if g.error_class == "timeout":
            n_slow += 1
            continue
        n_bad += 1
        print(f"       自评失败: {r['question_id']} {res.reason} {res.pred_error_class}")
    rep.check(n_bad == 0,
              f"gold 当预测全判对（错 {n_bad}/{len(sample)}；超时跳过 {n_slow} 条）")

    # 5/6/7. prompt 检查
    leak, truncated, missing_evidence = 0, 0, 0
    lens = []
    for r in rows:
        schema = reg.schema(r["db_id"])
        ddl = schema.render_ddl(BIRD_PROMPT.max_schema_cols,
                                include_types=BIRD_PROMPT.include_column_types,
                                include_fk=BIRD_PROMPT.include_fk,
                                quote_identifiers=BIRD_PROMPT.quote_identifiers)
        if len(ddl) > BIRD_PROMPT.max_schema_chars:
            truncated += 1
        msgs = build_messages(r["question"], schema, BIRD_PROMPT, evidence=r["evidence"])
        prompt = "\n".join(m["content"] for m in msgs)
        lens.append(len(prompt))
        gold = " ".join(str(r["query"]).split())
        if len(gold) > 30 and gold in " ".join(prompt.split()):
            leak += 1
        if r["evidence"] and r["evidence"] not in prompt:
            missing_evidence += 1
    rep.check(leak == 0, f"prompt 不含 gold SQL（泄漏 {leak} 条）")
    rep.check(truncated == 0, f"DDL 未被 {BIRD_PROMPT.max_schema_chars} 字符截断（{truncated} 条）")
    rep.check(missing_evidence == 0, f"有 evidence 的样本都拼进了 prompt（漏 {missing_evidence} 条）")
    rep.check(all(l < 20000 for l in lens), "prompt 长度没有异常膨胀")

    # 8. 与官方 JSON 交叉校验（条数 + SQL 一致）
    raw_path = ROOT / args.raw
    if raw_path.is_file():
        raw = load_bird_examples(raw_path)
        same = len(raw) == len(rows) + (0 if not args.limit else len(raw) - len(rows))
        rep.check(len(raw) == len(rows) or args.limit > 0,
                  f"与官方 dev.json 条数一致（官方 {len(raw)} / 本地 {len(rows)}）")
        if not args.limit:
            pairs = {r["question_id"]: r for r in raw}
            mismatch = sum(1 for r in rows
                           if pairs.get(r["question_id"], {}).get("query") != r["query"])
            rep.check(mismatch == 0, f"gold SQL 与官方逐条一致（不一致 {mismatch}）")

    print(f"\n[统计] prompt 字符：均值 {statistics.mean(lens):.0f}，"
          f"最大 {max(lens)}，≈{statistics.mean(lens)/3.5:.0f} tokens")
    print(f"[统计] 难度分布 {dict(diff)}")
    if rep.failed:
        print(f"\n[失败] {rep.failed} 项未通过")
        return 1
    print("\n[通过] BIRD 数据与评测口径自检全绿。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
