#!/usr/bin/env python3
"""BIRD 数据装配：把官方 dev/train 转成本项目训练与评测能直接吃的格式。

为什么需要这一步（和 P4 的 ``build_p4_data.py`` 同源）
------------------------------------------------------
BIRD 的原始 JSON 与 Spider 有三处不同，任何一处不处理都会在训练/评测时静默出错：

1. **字段名**：金标 SQL 叫 ``SQL``（大写），不是 ``query``；多一个 ``evidence``。
   → 在 :func:`envs.db_registry.load_bird_examples` 里统一抹平。
2. **库级切分**：BIRD 的 dev 只有 11 个库，train 有 69 个。RL 必须用 SFT 没见过的库，
   否则又是"训练库上的题模型已经会做"（P5 的零方差教训）。所以这里按 **db_id**
   把 train 切成 ``sft_dbs`` 与 ``rl_dbs``，两边零重叠，且都不含 dev 的库。
3. **prompt 形态**：BIRD 要带 evidence + 列类型 + 外键（见 ``rollout.prompt.BIRD_PROMPT``）。
   SFT 与 RL 的 prompt 必须**同一套**，所以都在这里预生成好，训练循环不再自己拼。

产出（默认 ``data/bird/``）
--------------------------
- ``dev.json``        评测样本（1534 条，字段与 Spider 的 dev_sample.json 同构）
- ``sft_train.jsonl`` 单轮 SFT：gold SQL 当答案（不花 API 钱，先拿到 BIRD 上的 SFT 底子）
- ``rl_train.jsonl``  RL prompt + gold（留出库；奖励用执行等价判定）
- ``db_split.json``   库级切分明细（可复现：``--seed`` 固定）
- ``meta.json``       统计 + 全部检查项

用法::

    python scripts/prepare_bird.py --dev-dir data/raw/bird/dev_20240627 \\
        --train-dir data/raw/bird/train --out-dir data/bird

    # 没有 train（只做评测）时：
    python scripts/prepare_bird.py --dev-dir ... --skip-train
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry, load_bird_examples  # noqa: E402
from envs.sql_sandbox import SqlSandbox  # noqa: E402
from rollout.prompt import BIRD_PROMPT, build_messages  # noqa: E402


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))]


def _length_stats(chars: list[int]) -> dict:
    if not chars:
        return {"n": 0}
    return {
        "n": len(chars),
        "chars_mean": round(statistics.mean(chars), 1),
        "chars_p95": _pct(chars, 0.95),
        "est_tokens_mean": round(statistics.mean(chars) / 3.5, 1),
        "est_tokens_p95": round(_pct(chars, 0.95) / 3.5, 1),
    }


def _gold_check_paths(rows: list[dict]) -> list[tuple[dict, bool, str]]:
    """对每条 gold 跑一次沙箱（timeout 由调用方给），返回执行结果。"""
    out = []
    for row in rows:
        sb = SqlSandbox(DBREG.db_path(row["db_id"]), timeout=2.0)
        res = sb.run(row["query"])
        out.append((row, res.success, res.error_class or ""))
    return out


DBREG: DatabaseRegistry  # 在 main 里赋值，供 _gold_check_paths 用


def main() -> int:
    global DBREG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", default="data/raw/bird/dev_20240627",
                    help="含 dev.json 与 dev_databases/ 的目录")
    ap.add_argument("--train-dir", default="data/raw/bird/train",
                    help="含 train.json 与 train_databases/ 的目录")
    ap.add_argument("--skip-train", action="store_true", help="只处理 dev（本地评测用）")
    ap.add_argument("--out-dir", default="data/bird")
    ap.add_argument("--holdout-dbs", type=int, default=15,
                    help="留给 RL 的库数（SFT 用剩下的；两边零重叠）")
    ap.add_argument("--sft-limit", type=int, default=0, help="SFT 条数上限（0=全用）")
    ap.add_argument("--rl-limit", type=int, default=0, help="RL 条数上限（0=全用）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-gold-check", action="store_true",
                    help="跳过 gold 逐条执行（1534 条约 50s，train 更久）")
    args = ap.parse_args()

    dev_dir = ROOT / args.dev_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    # ---------- 1. dev（评测集） ----------
    dev_rows = load_bird_examples(dev_dir / "dev.json")
    dev_reg = DatabaseRegistry(dev_dir / "dev_databases")
    DBREG = dev_reg
    print(f"[读取] dev {len(dev_rows)} 条 / {len({r['db_id'] for r in dev_rows})} 个库")

    dev_dbs = sorted({r["db_id"] for r in dev_rows})
    for db in dev_dbs:
        if not (dev_dir / "dev_databases" / db / f"{db}.sqlite").is_file():
            failures.append(f"dev 缺数据库文件: {db}")

    gold_ok = gold_timeout = 0
    if not args.skip_gold_check:
        for row, ok, err in _gold_check_paths(dev_rows):
            if ok:
                gold_ok += 1
            elif err == "timeout":
                gold_timeout += 1
        print(f"[检查] dev gold 可执行 {gold_ok}/{len(dev_rows)}（timeout {gold_timeout}）")
        if gold_ok + gold_timeout < len(dev_rows):
            failures.append(
                f"dev 有 {len(dev_rows) - gold_ok - gold_timeout} 条 gold 执行报错（非超时）")

    # dev 的 prompt 长度（BIRD preset：类型 + 外键 + evidence）
    dev_chars = []
    for row in dev_rows:
        schema = dev_reg.schema(row["db_id"])
        msgs = build_messages(row["question"], schema, BIRD_PROMPT, evidence=row["evidence"])
        dev_chars.append(sum(len(m["content"]) for m in msgs))
    dev_len = _length_stats(dev_chars)

    (out_dir / "dev.json").write_text(
        json.dumps(dev_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[输出] {out_dir.relative_to(ROOT)}/dev.json  {len(dev_rows)} 条")

    meta: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": {
            "include_column_types": BIRD_PROMPT.include_column_types,
            "include_fk": BIRD_PROMPT.include_fk,
            "include_evidence": BIRD_PROMPT.include_evidence,
            "quote_identifiers": BIRD_PROMPT.quote_identifiers,
            "max_schema_cols": BIRD_PROMPT.max_schema_cols,
        },
        "dev": {
            "n": len(dev_rows),
            "n_db": len(dev_dbs),
            "dbs": dev_dbs,
            "difficulty": dict(Counter(r["difficulty"] for r in dev_rows)),
            "n_evidence": sum(1 for r in dev_rows if r["evidence"]),
            "gold_ok": gold_ok,
            "gold_timeout": gold_timeout,
            "prompt_len": dev_len,
        },
    }

    # ---------- 2. train（SFT + RL，库级切分） ----------
    if not args.skip_train:
        train_dir = ROOT / args.train_dir
        train_json = train_dir / "train.json"
        if not train_json.is_file():
            failures.append(f"找不到 {train_json}")
        else:
            train_rows = load_bird_examples(train_json)
            train_reg = DatabaseRegistry(train_dir / "train_databases")
            DBREG = train_reg
            train_dbs = sorted({r["db_id"] for r in train_rows})
            overlap = sorted(set(train_dbs) & set(dev_dbs))
            if overlap:
                failures.append(f"train 与 dev 的库重叠（泄漏）: {overlap[:5]}")
            print(f"[读取] train {len(train_rows)} 条 / {len(train_dbs)} 个库")

            # 库级切分：先固定 seed 洗牌，再按 holdout_dbs 切
            shuffled = list(train_dbs)
            random.Random(args.seed).shuffle(shuffled)
            rl_dbs = sorted(shuffled[: args.holdout_dbs])
            sft_dbs = sorted(shuffled[args.holdout_dbs:])
            if not rl_dbs or not sft_dbs:
                failures.append(f"库级切分失败: sft={len(sft_dbs)} rl={len(rl_dbs)}")
            print(f"[切分] SFT 库 {len(sft_dbs)} / RL 库 {len(rl_dbs)}；"
                  f"RL 库={rl_dbs[:6]}{'...' if len(rl_dbs) > 6 else ''}")

            sft_rows_all = [r for r in train_rows if r["db_id"] in set(sft_dbs)]
            rl_rows_all = [r for r in train_rows if r["db_id"] in set(rl_dbs)]

            # SFT：单轮 messages（gold SQL 当答案），带 BIRD prompt
            sft_meta_reg = train_reg
            sft_out = []
            for r in sft_rows_all:
                schema = sft_meta_reg.schema(r["db_id"])
                msgs = build_messages(r["question"], schema, BIRD_PROMPT, evidence=r["evidence"])
                msgs.append({"role": "assistant", "content": f"```sql\n{r['query'].strip()}\n```"})
                sft_out.append({
                    "messages": msgs,
                    "meta": {
                        "db_id": r["db_id"],
                        "difficulty": r["difficulty"] or "unknown",
                        "n_turns": 1,
                        "split": "train",
                        "question": r["question"],
                        "question_id": r["question_id"],
                        "source": "bird:train_gold",
                        "prompt_chars": len(msgs[0]["content"]) + len(msgs[1]["content"]),
                    },
                })
            random.Random(args.seed + 1).shuffle(sft_out)
            if args.sft_limit:
                sft_out = sft_out[: args.sft_limit]

            # RL：prompt_messages + gold
            rl_out = []
            for i, r in enumerate(rl_rows_all):
                schema = train_reg.schema(r["db_id"])
                msgs = build_messages(r["question"], schema, BIRD_PROMPT, evidence=r["evidence"])
                rl_out.append({
                    "id": f"bird-rl-{i:05d}",
                    "db_id": r["db_id"],
                    "question": r["question"],
                    "query": r["query"],
                    "difficulty": r["difficulty"] or "unknown",
                    "question_id": r["question_id"],
                    "evidence": r["evidence"],
                    "prompt_messages": msgs,
                    "prompt_chars": len(msgs[0]["content"]) + len(msgs[1]["content"]),
                })
            random.Random(args.seed + 2).shuffle(rl_out)
            if args.rl_limit:
                rl_out = rl_out[: args.rl_limit]

            # 红线检查：SFT 与 RL 的 (db, question) 不许重叠
            # 注意 SFT 行的字段埋在 meta 里（训练用的格式），RL 行是平铺的。
            sft_keys = {(r["meta"]["db_id"], r["meta"]["question"]) for r in sft_out}
            dup = sum(1 for r in rl_out
                      if (r["db_id"], r["question"]) in sft_keys)
            if dup:
                failures.append(f"SFT 与 RL prompt 重叠 {dup} 条")
            if not sft_out or not rl_out:
                failures.append(f"数据量为空: sft={len(sft_out)} rl={len(rl_out)}")

            # SFT 再切 train/val：train_sft.py 必须要 sft_val.jsonl（用来算 val_loss）。
            # 按 2% 留出，且 val 的库仍属于 sft_dbs（不是 RL 库），不构成泄漏。
            n_val = max(20, int(len(sft_out) * 0.02))
            sft_val = sft_out[:n_val]
            sft_train = sft_out[n_val:]
            if not sft_val:
                failures.append("SFT val 切分为空")

            with (out_dir / "sft_train.jsonl").open("w", encoding="utf-8") as f:
                for it in sft_train:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            with (out_dir / "sft_val.jsonl").open("w", encoding="utf-8") as f:
                for it in sft_val:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            with (out_dir / "rl_train.jsonl").open("w", encoding="utf-8") as f:
                for it in rl_out:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            (out_dir / "db_split.json").write_text(json.dumps({
                "seed": args.seed,
                "sft_dbs": sft_dbs,
                "rl_dbs": rl_dbs,
                "dev_dbs": dev_dbs,
                "n_sft_train": len(sft_train),
                "n_sft_val": len(sft_val),
                "n_rl_rows": len(rl_out),
            }, ensure_ascii=False, indent=1), encoding="utf-8")

            meta["train"] = {
                "n": len(train_rows),
                "n_db": len(train_dbs),
                "n_sft_dbs": len(sft_dbs),
                "n_rl_dbs": len(rl_dbs),
                "n_sft_train": len(sft_train),
                "n_sft_val": len(sft_val),
                "n_rl": len(rl_out),
                "sft_difficulty": dict(Counter(r["meta"]["difficulty"] for r in sft_train)),
                "rl_difficulty": dict(Counter(r["difficulty"] for r in rl_out)),
                "sft_prompt_len": _length_stats([r["meta"]["prompt_chars"] for r in sft_train]),
                "rl_prompt_len": _length_stats([r["prompt_chars"] for r in rl_out]),
                "train_dev_db_overlap": len(set(train_dbs) & set(dev_dbs)),
                "sft_rl_prompt_overlap": dup,
                "difficulty_note": "BIRD 只有 dev.json 带 difficulty；train 的难度记为 unknown",
            }
            print(f"[输出] {out_dir.relative_to(ROOT)}/sft_train.jsonl  {len(sft_train)} 条")
            print(f"[输出] {out_dir.relative_to(ROOT)}/sft_val.jsonl    {len(sft_val)} 条")
            print(f"[输出] {out_dir.relative_to(ROOT)}/rl_train.jsonl   {len(rl_out)} 条")
            print(f"[统计] SFT 难度 {meta['train']['sft_difficulty']}")
            print(f"[统计] RL 难度  {meta['train']['rl_difficulty']}")

    meta["checks"] = {"ok": not failures, "failures": failures}
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[统计] dev prompt ≈{dev_len.get('est_tokens_mean')} tokens "
          f"(p95 {dev_len.get('est_tokens_p95')})")
    if failures:
        print("\n[失败] 数据检查未通过：")
        for f in failures:
            print("  -", f)
        return 1
    print("\n[通过] BIRD 数据装配完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
