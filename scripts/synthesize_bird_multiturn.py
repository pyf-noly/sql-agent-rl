#!/usr/bin/env python3
"""合成 BIRD 的多轮 SFT 轨迹：让强模型跑"判对才停"的 3 轮，只留成功轨迹。

为什么必须先做这一步（P6 的硬前置）
-----------------------------------
2026-11-05 的实测：BIRD-SFT 是**单轮**数据训出来的，三轮自修复只救回 9/153 条错题（5.9%），
也就是模型根本没学会"看反馈改 SQL"。轮级信用分配要能发力，先得让模型有多轮行为。

和 P3 的 ``synthesize_sft.py`` 的区别：
- 库、prompt、评测口径全部走 BIRD（evidence + 类型 + 外键 + 引号）；
- 轨迹用 ``messages_from_episode(ep, schema, BIRD_PROMPT)`` 还原，
  保证**训练序列与采样时逐字一致**（这是 P4 踩过的坑）；
- 带 ``--max-cost`` 硬上限：每提交一批前检查已花费，超了就停（租不起第二张卡）。

产物
----
``data/bird/episodes_multiturn.jsonl``   原始轨迹（可断点续跑，带 set_id 指纹）
``data/bird/sft_multiturn.jsonl``        多轮 SFT 样本（只含成功轨迹，messages 已带 schema）
``data/bird/multiturn.meta.json``        统计：可救率、轮数分布、花费

用法::

    python scripts/synthesize_bird_multiturn.py --dry-run 3         # 先花几分钱看看
    python scripts/synthesize_bird_multiturn.py --limit 1200 --max-cost 12
"""

from __future__ import annotations

import argparse
import hashlib
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
from rollout.evaluator import Evaluator  # noqa: E402
from rollout.llm_client import make_client_from_env  # noqa: E402
from rollout.prompt import BIRD_PROMPT  # noqa: E402
from training.chatfmt import messages_from_episode  # noqa: E402


def fingerprint(seeds: list[dict]) -> str:
    payload = json.dumps([(s["db_id"], s["question"]) for s in seeds],
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_done(path: Path, set_id: str) -> set[int]:
    done: set[int] = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("set_id") == set_id:
                done.add(d["idx"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", default="data/raw/bird/train/train.json")
    ap.add_argument("--db-root", default="data/raw/bird/train/train_databases")
    ap.add_argument("--split", default="data/bird/db_split.json")
    ap.add_argument("--out-dir", default="data/bird")
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-cost", type=float, default=12.0, help="人民币硬上限")
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_bird_examples(ROOT / args.train_json)
    split_p = ROOT / args.split
    if split_p.is_file():
        sft_dbs = set(json.loads(split_p.read_text(encoding="utf-8")).get("sft_dbs") or [])
        if sft_dbs:
            rows = [r for r in rows if r["db_id"] in sft_dbs]
            print(f"[过滤] 只用 SFT 库（{len(sft_dbs)} 个）→ {len(rows)} 条")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.dry_run:
        rows = rows[: args.dry_run]
    elif args.limit:
        rows = rows[: args.limit]
    print(f"[种子] {len(rows)} 条")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_path = out_dir / "episodes_multiturn.jsonl"
    set_id = fingerprint(rows)
    done = load_done(ep_path, set_id)
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"[续跑] 已完成 {len(done)}，待跑 {len(todo)}，种子集 {set_id}")

    registry = DatabaseRegistry(ROOT / args.db_root)
    client = make_client_from_env(temperature=args.temperature, max_tokens=args.max_tokens)
    ev = Evaluator(registry, timeout=10.0, max_rows=20000, compare="bird")
    agent = SqlAgent(
        client, registry, max_turns=args.turns, timeout=10.0,
        temperature=args.temperature,
        stop_on_success=False, verifier=ev, stop_on_correct=True,
        prompt_cfg=BIRD_PROMPT, max_rows=20000,
    )

    t0 = time.time()
    n_ok = n_run = 0
    stopped = False
    turns_sum = 0
    with open(ep_path, "a", encoding="utf-8") as fout:
        for i in todo:
            if client.usage.cost_cny >= args.max_cost:
                print(f"\n[停止] 已花费 ¥{client.usage.cost_cny:.2f} ≥ 上限 ¥{args.max_cost}")
                stopped = True
                break
            r = rows[i]
            try:
                ep = agent.run(r)
            except Exception as e:  # noqa: BLE001
                print(f"\n[异常] #{i}: {type(e).__name__}: {e}")
                continue
            n_run += 1
            ok = any(t.correct for t in ep.turns)
            n_ok += int(ok)
            turns_sum += ep.n_turns
            d = ep.to_dict(truncate_feedback=False)
            # 关键：messages 用**训练同一个函数**从内存里的 Episode 直接还原
            # （Turn.feedback_message 不在 to_dict 里，序列化后就丢了）
            try:
                d["messages"] = messages_from_episode(ep, registry.schema(r["db_id"]), BIRD_PROMPT)
            except Exception as e:  # noqa: BLE001
                print(f"\n[警告] #{i} 还原 messages 失败: {type(e).__name__}: {e}")
                d["messages"] = None
            d.update(idx=i, set_id=set_id, difficulty=r.get("difficulty"),
                     db_id=r["db_id"], evidence=r.get("evidence", ""), correct=ok,
                     question_id=r.get("question_id"))
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            fout.flush()
            if n_run % 10 == 0:
                print(f"\r      {n_run}/{len(todo)}  可救率 {n_ok/n_run:.0%}  "
                      f"平均轮数 {turns_sum/n_run:.2f}  "
                      f"花费 ¥{client.usage.cost_cny:.2f}", end="", flush=True)
    print()
    print(f"[合成] 跑 {n_run} 条，成功 {n_ok} 条，耗时 {time.time()-t0:.0f}s，"
          f"花费 ¥{client.usage.cost_cny:.3f}")

    # ---- 转 SFT 样本（messages 已在合成时用训练同一个函数生成）----
    sft_path = out_dir / "sft_multiturn.jsonl"
    n_sft = n_multi = 0
    turn_hist: Counter = Counter()
    with open(ep_path, encoding="utf-8") as fin, open(sft_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("set_id") != set_id or not d.get("correct") or not d.get("messages"):
                continue
            n_turns = len(d.get("turns") or [])
            turn_hist[n_turns] += 1
            fout.write(json.dumps({
                "messages": d["messages"],
                "meta": {
                    "db_id": d["db_id"], "difficulty": d.get("difficulty") or "unknown",
                    "n_turns": n_turns, "split": "train",
                    "question": d.get("question"), "question_id": d.get("question_id"),
                    "source": "bird:multiturn_synth",
                    "prompt_chars": sum(len(m["content"]) for m in d["messages"]),
                },
            }, ensure_ascii=False) + "\n")
            n_sft += 1
            n_multi += int(n_turns >= 2)
    print(f"[SFT] {n_sft} 条（多轮 {n_multi} / 单轮 {n_sft-n_multi}）→ {sft_path.relative_to(ROOT)}")
    print(f"[统计] 轮数分布 {dict(sorted(turn_hist.items()))}")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "set_id": set_id,
        "n_seeds": len(rows), "n_run": n_run, "n_ok": n_ok,
        "rescue_rate": round(n_ok / max(1, n_run), 4),
        "n_sft": n_sft, "n_multi": n_multi,
        "turns_hist": dict(sorted(turn_hist.items())),
        "max_turns": args.turns, "temperature": args.temperature,
        "cost_cny": round(client.usage.cost_cny, 4),
        "usage": client.usage.to_dict(),
        "early_stopped": stopped,
        "prompt_preset": "bird",
    }
    (out_dir / "multiturn.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[花费] ¥{client.usage.cost_cny:.3f}  缓存命中 {client.usage.cache_hits}")
    return 0


def _rebuild_episode(d: dict):
    """把 episodes.jsonl 的一行还原成 Episode（只填 SFT 需要的字段）。"""
    from rollout.agent import Episode, Turn

    ep = Episode(db_id=d["db_id"], question=d["question"], gold_sql=d["gold_sql"],
                 evidence=d.get("evidence", ""))
    for t in d["turns"]:
        ep.turns.append(Turn(
            turn=t["turn"], sql=t.get("sql", ""), success=bool(t.get("success")),
            error_class=t.get("error_class"), error_message=t.get("error_message", ""),
            row_count=t.get("row_count", 0), parse_source=t.get("parse_source", ""),
            feedback=t.get("feedback", ""),
            feedback_message=({"role": "user", "content": t["feedback_message"]}
                              if t.get("feedback_message") else None),
            correct=t.get("correct"),
        ))
    ep.final_sql = d.get("final_sql", "")
    ep.stopped_reason = d.get("stopped_reason", "")
    return ep


if __name__ == "__main__":
    sys.exit(main())
