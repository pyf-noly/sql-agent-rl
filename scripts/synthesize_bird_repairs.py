#!/usr/bin/env python3
"""把本地 4B 的错误尝试交给强模型"看一眼反馈再改"，合成多轮修复轨迹。

和 P3 的 ``synthesize_sft.py`` 的区别：**第一轮不是强模型生成的，而是我们自己模型的
真实错误**（来自 ``gen_bird_attempts.py``）。这样得到的轨迹：

    用户: schema + evidence + 问题
    助手: <4B 的真实错误 SQL>          ← 线上模型就会这么错
    用户: 真实执行反馈（错在哪、候选列名……）
    助手: <强模型看到反馈后的正确 SQL>  ← 要教的是这个

比"让强模型从头做"强在哪：强模型一次就做对（实测 3/3 首轮正确），
根本没有修复过程可学；而用真实错误当起点，修复过程一定存在，且错误分布与线上一致。

产物
----
``data/bird/repairs.jsonl``      每条成功修复的轨迹（含 messages，可直接进 SFT）
``data/bird/repairs.meta.json``  统计 + 花费

用法::

    python scripts/synthesize_bird_repairs.py --limit 20 --dry-run   # 几分钱
    python scripts/synthesize_bird_repairs.py --limit 600 --max-cost 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402
from rollout.llm_client import make_client_from_env  # noqa: E402
from rollout.parse_sql import parse_sql  # noqa: E402
from rollout.prompt import BIRD_PROMPT, build_feedback_message, build_messages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", default="data/bird/attempts.jsonl")
    ap.add_argument("--db-root", default="data/raw/bird/train/train_databases")
    ap.add_argument("--out", default="data/bird/repairs.jsonl")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-cost", type=float, default=6.0)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--only-detectable", action="store_true", default=True,
                    help="只修复反馈里能看出错的样本（报错/超时/空结果）；"
                         "执行成功但结果错的样本教师没有依据去改，实测成功率仅 10%")
    ap.add_argument("--include-semantic", action="store_true",
                    help="也尝试修复执行成功但结果错的样本（默认关，见上）")
    args = ap.parse_args()
    if args.include_semantic:
        args.only_detectable = False

    att_path = ROOT / args.attempts
    if not att_path.is_file():
        print(f"[失败] 找不到 {att_path}，先跑 gen_bird_attempts.py")
        return 1
    rows = [json.loads(line) for line in att_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    wrong = [r for r in rows if not r.get("correct") and r.get("sql")]
    print(f"[输入] 尝试 {len(rows)} 条，其中错误 {len(wrong)} 条")

    # 关键筛选：只有"反馈里能看出错"的样本才适合做修复训练。
    # 实测：4B 的错误里约一半是"执行成功但结果不对"（比如 join 错了字段），
    # 反馈只有 "Execution succeeded: 1 row"，模型没有任何依据去改——
    # 让教师在这种反馈下"修复"，成功率只有 10%，而且学到的是瞎猜。
    # 所以只保留可检测的错误：执行报错 / 语法错 / 超时 / 结果为空。
    if args.only_detectable:
        det = []
        for r in wrong:
            ec = r.get("error_class")
            if ec in ("db_error", "invalid_sql", "timeout") or r.get("row_count") == 0:
                det.append(r)
        print(f"[筛选] 可检测错误（报错/超时/空结果）{len(det)}/{len(wrong)} 条")
        wrong = det

    out_path = ROOT / args.out
    done_qids = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done_qids.add(json.loads(line)["question_id"])
            except Exception:  # noqa: BLE001
                continue
    todo = [r for r in wrong if r.get("question_id") not in done_qids]
    if args.dry_run:
        todo = todo[: args.dry_run]
    elif args.limit:
        todo = todo[: args.limit]
    print(f"[计划] 待修复 {len(todo)} 条（已完成 {len(done_qids)}）")

    registry = DatabaseRegistry(ROOT / args.db_root)
    ev = Evaluator(registry, timeout=args.timeout, max_rows=20000, compare="bird")
    client = make_client_from_env(temperature=args.temperature, max_tokens=args.max_tokens)

    t0 = time.time()
    n_ok = n_run = 0
    turns = Counter()
    stopped = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fout:
        for k, r in enumerate(todo, 1):
            if client.usage.cost_cny >= args.max_cost:
                print(f"\n[停止] 已花费 ¥{client.usage.cost_cny:.2f} ≥ 上限 ¥{args.max_cost}")
                stopped = True
                break
            schema = registry.schema(r["db_id"])
            base_msgs = build_messages(r["question"], schema, BIRD_PROMPT,
                                       evidence=r.get("evidence") or "")
            # 第 1 轮：本地模型的真实错误输出（原文，不重写）
            first = r.get("raw_output") or f"```sql\n{r['sql']}\n```"
            fb_msg = build_feedback_message(r["sql"], r.get("feedback") or "")
            messages = base_msgs + [{"role": "assistant", "content": first}, fb_msg]
            try:
                text = client.chat(messages)
            except Exception as e:  # noqa: BLE001
                print(f"\n[异常] #{r.get('question_id')}: {type(e).__name__}: {e}")
                continue
            n_run += 1
            parsed = parse_sql(text)
            ok = False
            reason = "parse_error"
            if parsed.ok:
                res = ev.evaluate(r["db_id"], r["gold_sql"], parsed.sql)
                ok = res.correct
                reason = res.reason
            n_ok += int(ok)
            if ok:
                turns[2] += 1
                fout.write(json.dumps({
                    "question_id": r.get("question_id"),
                    "db_id": r["db_id"],
                    "question": r["question"],
                    "evidence": r.get("evidence", ""),
                    "first_sql": r["sql"],
                    "first_error_class": r.get("error_class"),
                    "fixed_sql": parsed.sql,
                    "messages": messages + [{"role": "assistant",
                                             "content": f"```sql\n{parsed.sql}\n```"}],
                }, ensure_ascii=False) + "\n")
                fout.flush()
            if n_run % 10 == 0:
                print(f"\r      {n_run}/{len(todo)}  修复成功率 {n_ok/n_run:.0%}  "
                      f"花费 ¥{client.usage.cost_cny:.2f}", end="", flush=True)

    print()
    print(f"[完成] 修复 {n_ok}/{n_run} 条，耗时 {(time.time()-t0)/60:.1f} min，"
          f"花费 ¥{client.usage.cost_cny:.3f}")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_input_wrong": len(wrong), "n_run": n_run, "n_ok": n_ok,
        "repair_rate": round(n_ok / max(1, n_run), 4),
        "cost_cny": round(client.usage.cost_cny, 4),
        "usage": client.usage.to_dict(),
        "early_stopped": stopped,
        "model": client.model,
    }
    (out_path.parent / "repairs.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
