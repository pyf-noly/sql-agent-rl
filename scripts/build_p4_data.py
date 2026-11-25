#!/usr/bin/env python3
"""P4 数据装配：把 P3 的产物转成训练循环能直接吃的三份文件。

为什么不能直接拿 ``sft_final.jsonl`` 去训练
--------------------------------------------
审计 P3 产物时发现三个**会导致训练-推理不一致**的问题，全部在这一步修掉：

1. **SFT 数据没有 system prompt 和 schema**。
   ``sft_final.jsonl`` 的 messages 是 ``[user: 原问题, assistant: ```sql ...```]``，
   而推理时模型看到的是 ``[system: 你是 SQLite 专家..., user: schema + 问题 + 输出要求]``。
   拿前者训练，模型学到的是"不看库表直接编 SQL"，上线必崩。

2. **多轮样本的反馈消息格式与推理时不一致**。
   SFT 里第 2 轮的用户消息只有 ``Execution succeeded: 30 rows...``，
   而推理时是 ``Your previous attempt:\\n```sql ...```\\n\\n<反馈>\\n\\n请改正``。
   训练和推理的输入分布不一致，等于让模型在考试时看一份没见过的卷子。

3. **RL prompt 集不含 gold**（``rl_prompts.json`` 只有 db_id/question/difficulty），
   但 GRPO 的奖励必须执行等价判定，没有 gold 就算不出奖励。
   好在这些样本全部来自 Spider train，回查 ``train_spider.json`` 即可无损补回。

这三件事都**不需要 GPU**，也必须在上卡之前做完——否则你会在租来的机器上
花着 ¥2/小时排查一个纯数据格式 bug。

产出（默认 ``data/train/``）
---------------------------
- ``sft_train.jsonl`` / ``sft_val.jsonl``：推理同构的 messages + 每条样本的 token 长度标记来源
- ``rl_train.jsonl``：RL prompt + gold SQL（奖励函数要用）
- ``train_data.meta.json``：统计 + 全部检查项的结果（谁失败谁退出码非 0）

用法::

    python scripts/build_p4_data.py                 # 全量
    python scripts/build_p4_data.py --limit 50      # 冒烟用
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry, load_spider_examples  # noqa: E402
from rollout.prompt import build_feedback_message, build_messages  # noqa: E402

#: P3 的反馈文本是按 500 字符截断后存的（``Episode.to_dict`` 的默认行为）。
#: 正好 500 字符的样本高度可疑，单独计数并在报告里提示。
TRUNCATION_HINT = 500

_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.S | re.I)


def sql_block(sql: str) -> str:
    """把裸 SQL 包成推理时要求的 ```sql 代码块（与 P1 的解析器约定一致）。"""
    return f"```sql\n{sql.strip()}\n```"


def extract_sql(text: str) -> str:
    """从助手回复里抽 SQL；抽不到就原样返回（用于重建"上一轮 SQL"）。"""
    m = _SQL_BLOCK_RE.search(text or "")
    return (m.group(1).strip() if m else (text or "").strip())


def rebuild_sft_messages(rec: dict, registry: DatabaseRegistry) -> list[dict]:
    """把 P3 的 ``[问题, SQL, 反馈, SQL, ...]`` 重建成推理同构的 messages。

    重建规则（与 ``rollout/agent.py`` 完全一致）::

        system  : SYSTEM_PROMPT（来自 rollout.prompt，**不复制**，避免两套模板漂移）
        user    : build_user_prompt(问题, schema)
        assistant: 模型原始输出（保留原文，包括可能的解释性文字）
        user    : build_feedback_message(上一轮 SQL, 反馈文本)
        ...

    只做"加壳"：助手内容一个字不改，反馈文本一个字不改。
    """
    meta = rec["meta"]
    db_id = meta["db_id"]
    schema = registry.schema(db_id)
    old = rec["messages"]
    if not old or old[0]["role"] != "user":
        raise ValueError(f"意外的消息结构: {[m['role'] for m in old]}")

    out = build_messages(old[0]["content"], schema)
    prev_sql = ""
    for m in old[1:]:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": m["content"]})
            prev_sql = extract_sql(m["content"])
        elif m["role"] == "user":
            if not prev_sql:
                raise ValueError("反馈消息之前没有 assistant 轮次")
            out.append(build_feedback_message(prev_sql, m["content"]))
        else:  # pragma: no cover - P3 只产生这两种角色
            raise ValueError(f"未知角色 {m['role']}")
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_rl_with_gold(rl_path: Path, spider_train: Path) -> tuple[list[dict], dict]:
    """给 RL prompt 回填 gold SQL。

    匹配键是 ``(db_id, question)``。Spider 里同一问题可能在多个库出现，
    所以 db_id 必须参与匹配；匹配不上直接报错，不许静默丢弃。
    """
    prompts = json.loads(rl_path.read_text(encoding="utf-8"))
    spider = load_spider_examples(spider_train)
    index: dict[tuple[str, str], list[str]] = {}
    for ex in spider:
        index.setdefault((ex["db_id"], ex["question"]), []).append(ex["query"])

    rows: list[dict] = []
    missing: list[dict] = []
    multi: list[str] = []
    for p in prompts:
        key = (p["db_id"], p["question"])
        hits = index.get(key, [])
        if not hits:
            missing.append(p)
            continue
        if len(hits) > 1:
            multi.append(p["question"])
        rows.append({**p, "query": hits[0]})
    stats = {"n_missing": len(missing), "n_multi_gold": len(multi),
             "missing_examples": missing[:3]}
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/processed/sft_final.jsonl")
    ap.add_argument("--rl", default="data/processed/rl_prompts.json")
    ap.add_argument("--spider-train", default="data/raw/spider/train_spider.json")
    ap.add_argument("--spider-dev", default="data/raw/spider/dev.json")
    ap.add_argument("--db-root", default="data/raw/spider/database")
    ap.add_argument("--out-dir", default="data/train")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟）")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = DatabaseRegistry(ROOT / args.db_root)

    sft_recs = [json.loads(line) for line in
                (ROOT / args.sft).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        sft_recs = sft_recs[: args.limit]
    print(f"[读取] SFT {len(sft_recs)} 条，{args.sft}")

    # ---------- 1. SFT 重建 ----------
    train: list[dict] = []
    val: list[dict] = []
    sft_errors: list[str] = []
    fb_hits_500 = 0
    db_missing: set[str] = set()

    for i, rec in enumerate(sft_recs):
        meta = rec["meta"]
        try:
            messages = rebuild_sft_messages(rec, registry)
        except FileNotFoundError:
            db_missing.add(meta["db_id"])
            continue
        except Exception as e:  # noqa: BLE001
            sft_errors.append(f"#{i} {meta.get('db_id')}: {type(e).__name__}: {e}")
            continue

        for m in messages[2:]:
            if m["role"] == "user" and len(m["content"]) >= TRUNCATION_HINT:
                fb_hits_500 += 1

        item = {
            "messages": messages,
            "meta": {
                "db_id": meta["db_id"],
                "difficulty": meta.get("difficulty"),
                "n_turns": meta.get("n_turns", sum(m["role"] == "assistant" for m in messages)),
                "split": meta.get("split", "train"),
                "question": rec["messages"][0]["content"],
                "source": "p3:sft_final",
            },
        }
        # 用 schema 字符数做长度代理，训练时按真实 token 长度再分桶
        item["meta"]["prompt_chars"] = len(messages[1]["content"]) + len(messages[0]["content"])
        (val if meta.get("split") == "val" else train).append(item)

    # ---------- 2. RL 回填 gold ----------
    rl_rows, rl_stats = load_rl_with_gold(ROOT / args.rl, ROOT / args.spider_train)
    if args.limit:
        rl_rows = rl_rows[: args.limit]

    rl_out: list[dict] = []
    for i, r in enumerate(rl_rows):
        schema = registry.schema(r["db_id"])
        msgs = build_messages(r["question"], schema)
        rl_out.append({
            "id": f"rl-{i:05d}",
            "db_id": r["db_id"],
            "question": r["question"],
            "query": r["query"],
            "difficulty": r.get("difficulty"),
            "prompt_messages": msgs,
            "prompt_chars": len(msgs[0]["content"]) + len(msgs[1]["content"]),
        })

    # ---------- 3. 检查 ----------
    dev_ids = {x["db_id"] for x in load_spider_examples(ROOT / args.spider_dev)}
    rl_ids = {r["db_id"] for r in rl_out}
    sft_train_ids = {r["meta"]["db_id"] for r in train}

    failures: list[str] = []
    if sft_errors:
        failures.append(f"SFT 重建失败 {len(sft_errors)} 条: {sft_errors[:3]}")
    if db_missing:
        failures.append(f"缺数据库文件: {sorted(db_missing)[:5]}")
    if rl_stats["n_missing"]:
        failures.append(f"RL 样本回填不到 gold {rl_stats['n_missing']} 条")
    leak_ids = sorted(rl_ids & dev_ids)
    if leak_ids:
        failures.append(f"RL 训练集与 dev 评测集有 {len(leak_ids)} 个库重叠（泄漏）: {leak_ids[:5]}")
    if not train or not val:
        failures.append(f"切分异常: train={len(train)} val={len(val)}")

    # 抽样做"格式 + 不许泄漏答案"的硬检查
    for item in (train[:200] + val[:50]):
        msgs = item["messages"]
        if msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
            failures.append("messages 前两条必须是 system/user")
            break
        if "```sql" not in msgs[1]["content"]:
            failures.append("user 消息里没有 schema 的 ```sql 块")
            break
        assists = [m for m in msgs if m["role"] == "assistant"]
        if not assists or any("```sql" not in a["content"] for a in assists):
            failures.append("assistant 消息里没有 ```sql 块")
            break

    # 第二条红线：反馈里不许出现 gold SQL（否则等于开卷考试）
    spider_query = {(x["db_id"], x["question"]): x["query"]
                    for x in load_spider_examples(ROOT / args.spider_train)}
    leaked = 0
    for item in (train[:500] + val[:100]):
        gold = _norm(spider_query.get(
            (item["meta"]["db_id"], item["meta"]["question"]), ""))
        if len(gold) < 20:
            continue
        for m in item["messages"][2:]:
            if m["role"] == "user" and gold in _norm(m["content"]):
                leaked += 1
    if leaked:
        failures.append(f"反馈文本里出现 gold SQL（泄漏）: {leaked} 处")

    if any(not r["query"].strip() for r in rl_out):
        failures.append("RL 数据里有空的 gold")

    # ---------- 4. 统计与落盘 ----------
    def _chars(items: list[dict]) -> dict:
        vals = [sum(len(m["content"]) for m in it["messages"]) for it in items]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "chars_mean": round(statistics.mean(vals), 1),
            "chars_p95": sorted(vals)[int(len(vals) * 0.95) - 1],
            "chars_max": max(vals),
            "est_tokens_mean": round(statistics.mean(vals) / 3.5, 1),
            "est_tokens_p95": round(sorted(vals)[int(len(vals) * 0.95) - 1] / 3.5, 1),
        }

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {"sft": args.sft, "rl": args.rl, "spider_train": args.spider_train},
        "n_sft_train": len(train),
        "n_sft_val": len(val),
        "n_rl": len(rl_out),
        "sft_turns": dict(Counter(it["meta"]["n_turns"] for it in train)),
        "sft_difficulty": dict(Counter(it["meta"]["difficulty"] for it in train)),
        "rl_difficulty": dict(Counter(r["difficulty"] for r in rl_out)),
        "n_db_sft": len(sft_train_ids),
        "n_db_rl": len(rl_ids),
        "n_db_dev": len(dev_ids),
        "rl_dev_db_overlap": len(rl_ids & dev_ids),
        "rl_multi_gold": rl_stats["n_multi_gold"],
        "feedback_truncation_suspicious": fb_hits_500,
        "length_sft_train": _chars(train),
        "length_rl_prompt": _chars([{"messages": r["prompt_messages"]} for r in rl_out]) if rl_out else {"n": 0},
        "checks": {"ok": not failures, "failures": failures},
    }

    with (out_dir / "sft_train.jsonl").open("w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out_dir / "sft_val.jsonl").open("w", encoding="utf-8") as f:
        for it in val:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out_dir / "rl_train.jsonl").open("w", encoding="utf-8") as f:
        for r in rl_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "train_data.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 5. 打印 ----------
    print(f"[输出] {out_dir.relative_to(ROOT)}/sft_train.jsonl  {len(train)} 条")
    print(f"[输出] {out_dir.relative_to(ROOT)}/sft_val.jsonl    {len(val)} 条")
    print(f"[输出] {out_dir.relative_to(ROOT)}/rl_train.jsonl   {len(rl_out)} 条")
    print(f"[统计] SFT 轮数分布 {meta['sft_turns']}，难度 {meta['sft_difficulty']}")
    print(f"[统计] 长度（字符）均值 {meta['length_sft_train'].get('chars_mean')}，"
          f"p95 {meta['length_sft_train'].get('chars_p95')}，"
          f"≈{meta['length_sft_train'].get('est_tokens_mean')} tokens")
    print(f"[统计] RL 与 dev 库重叠 {meta['rl_dev_db_overlap']}（必须为 0）")
    print(f"[统计] 反馈疑似被 500 字符截断 {fb_hits_500} 条")
    if failures:
        print("\n[失败] 数据检查未通过：")
        for f in failures:
            print("  -", f)
        return 1
    print("\n[通过] 全部检查项通过，可以上卡。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
