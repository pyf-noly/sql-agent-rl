#!/usr/bin/env python3
"""合成 SFT 冷启动轨迹：让强模型跑"判对才停"的多轮，只留成功的轨迹。

**这是 P3 唯一的收费环节**（约 ¥5~15）。所以：

1. 先 ``--dry-run 3`` 跑 3 条，看轨迹长什么样、花多少钱；
2. 确认没问题再 ``--limit 0`` 跑全量；
3. 支持断点续跑（已完成的样本跳过）。

产物
----
``data/synthetic/episodes.jsonl``
    每行一个 Episode（含全部轮次、反馈、是否判对）。
``data/processed/sft_coldstart.jsonl``
    清洗后的 SFT 训练样本：``{"messages": [...], "meta": {...}}``。
    只保留"最终判对"的轨迹；格式是 chat 多轮，带执行反馈。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.agent import SqlAgent  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402
from rollout.llm_client import make_client_from_env  # noqa: E402


def _seed_fingerprint(seeds: list[dict]) -> str:
    """给当前种子集算一个短指纹。

    断点续跑必须按 ``(指纹, idx)`` 判断，不能用全局 ``idx``——
    换了 ``--only-difficulty`` / ``--limit`` 之后 idx 会重叠，
    会把别的批次的进度误当成本批已完成（这个坑真实踩过）。
    """
    payload = json.dumps(
        [(s.get("db_id"), s.get("question"), s.get("difficulty")) for s in seeds],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _load_done(path: Path, fingerprint: str) -> tuple[set[int], int]:
    """只把属于当前种子集的记录算作"已完成"。

    返回 ``(已完成的 idx, 无 set_id 的遗留记录数)``。
    遗留记录（旧版本脚本写的，没有 set_id）在本次运行时会被**重新跑一遍**，
    因为无法证明它对应哪个种子集——宁可多花几分钱，也不要拿错数据训练。
    """
    done: set[int] = set()
    legacy = 0
    if not path.is_file():
        return done, legacy
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            sid = d.get("set_id")
            if sid is None:
                legacy += 1
                continue
            if sid == fingerprint:
                done.add(d["idx"])
    return done, legacy


def _to_sft(ep_dict: dict) -> dict | None:
    """把一条成功轨迹转成 chat 格式的 SFT 样本。

    注意：保留**中间失败轮 + 反馈**，因为要教的不只是"写对 SQL"，
    还有"看到报错怎么改"——这正是 agentic 能力的来源。
    """
    turns = ep_dict.get("turns") or []
    if not turns:
        return None
    if not any(t.get("correct") for t in turns):
        return None

    messages: list[dict] = []
    for i, t in enumerate(turns):
        if i == 0:
            # 首轮用户消息：问题 + schema 由训练时的 collator 拼接，
            # 这里只存问题，避免把 schema 重复存几千遍。
            messages.append({"role": "user", "content": ep_dict.get("question", "")})
        sql = (t.get("sql") or "").strip()
        if not sql:
            continue
        messages.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})
        if i == len(turns) - 1:
            break
        fb = t.get("feedback") or ""
        if fb:
            messages.append({"role": "user", "content": fb})
    if len(messages) < 2:
        return None
    return {
        "messages": messages,
        "meta": {
            "db_id": ep_dict.get("db_id"),
            "difficulty": ep_dict.get("difficulty"),
            "n_turns": len(turns),
            "stopped_reason": ep_dict.get("stopped_reason"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/processed/sft_seeds.json")
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--dry-run", type=int, default=0, help="只跑 N 条看看")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="合成时用较高温度增加多样性")
    ap.add_argument("--only-difficulty", default="",
                    help="只合成指定难度，逗号分隔，如 hard,extra（省时省钱且对准多轮目标）")
    ap.add_argument("--sft-mode", choices=["all", "multi-turns-only"], default="all",
                    help="multi-turns-only: 只保留真的经历过多轮修正的轨迹")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="保留重复题（默认去重，避免 idx 错位）")
    args = ap.parse_args()

    seeds_path = ROOT / args.seeds
    if not seeds_path.is_file():
        print(f"[失败] 找不到 {seeds_path}，先跑 prep_p3_data.py")
        return 1
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    if args.only_difficulty:
        want = {d.strip() for d in args.only_difficulty.split(",") if d.strip()}
        before = len(seeds)
        seeds = [s for s in seeds if s.get("difficulty") in want]
        print(f"[筛选] 难度 {sorted(want)}：{before} -> {len(seeds)} 条")

    # 去重：Spider train 里存在完全相同的 (db_id, question)。
    # 不去重会导致 idx 与内容不是一一对应，断点续跑和回填都会错位。
    if not args.keep_duplicates:
        seen: set[tuple] = set()
        uniq: list[dict] = []
        for s in seeds:
            k = (s.get("db_id"), s.get("question"))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(s)
        if len(uniq) != len(seeds):
            print(f"[去重] {len(seeds)} -> {len(uniq)}（移除 {len(seeds)-len(uniq)} 条重复题）")
        seeds = uniq

    if args.dry_run:
        seeds = seeds[: args.dry_run]
    elif args.limit:
        seeds = seeds[: args.limit]
    print(f"[种子] {len(seeds)} 条")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_path = out_dir / "episodes.jsonl"
    set_id = _seed_fingerprint(seeds)
    print(f"[种子集] id={set_id}")
    done, legacy = _load_done(ep_path, set_id)
    todo = [i for i in range(len(seeds)) if i not in done]
    print(f"[续跑] 本批已完成 {len(done)}，本次待跑 {len(todo)}")
    if legacy:
        print(f"[提示] 另有 {legacy} 条旧记录缺少 set_id（无法确认属于哪个种子集），"
              f"本次会重跑覆盖")

    if not todo:
        print("全部已完成，跳过合成")
    else:
        registry = DatabaseRegistry(ROOT / "data/raw/spider/database")
        client = make_client_from_env(temperature=args.temperature, max_tokens=512)
        ev = Evaluator(registry)
        agent = SqlAgent(
            client, registry, max_turns=args.turns,
            stop_on_success=False, verifier=ev, stop_on_correct=True,
        )

        t0 = time.time()
        n_ok = 0
        with open(ep_path, "a", encoding="utf-8") as fout, \
                ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(agent.run, seeds[i]): i for i in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                try:
                    ep = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"\n[异常] #{i}: {type(e).__name__}: {e}")
                    continue
                d = ep.to_dict(truncate_feedback=False)
                d["idx"] = i
                d["set_id"] = set_id
                d["difficulty"] = seeds[i].get("difficulty")
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                fout.flush()
                if any(t.get("correct") for t in d["turns"]):
                    n_ok += 1
                if k % 10 == 0:
                    rate = n_ok / k
                    print(f"\r      完成 {k}/{len(todo)}  可救率 {rate:.0%}  "
                          f"花费 ¥{client.usage.cost_cny:.2f}", end="", flush=True)

        print()
        print(f"[合成] 成功轨迹 {n_ok}/{len(todo)}，"
              f"花费 ¥{client.usage.cost_cny:.3f}，耗时 {time.time()-t0:.0f}s")

    # 清洗成 SFT 格式
    sft_path = ROOT / "data/processed/sft_coldstart.jsonl"
    n_written = 0
    n_single = n_multi = 0
    with open(ep_path, encoding="utf-8") as fin, \
            open(sft_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            item = _to_sft(d)
            if not item:
                continue
            n_turns = item["meta"]["n_turns"]
            if args.sft_mode == "multi-turns-only" and n_turns < 2:
                continue
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_written += 1
            if n_turns >= 2:
                n_multi += 1
            else:
                n_single += 1
    print(f"[清洗] SFT 样本 {n_written} 条 -> {sft_path.relative_to(ROOT)}"
          f"（单轮 {n_single} / 多轮 {n_multi}）")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": str(seeds_path.relative_to(ROOT)),
        "n_episodes": sum(1 for _ in open(ep_path, encoding="utf-8")) if ep_path.is_file() else 0,
        "n_sft": n_written,
        "max_turns": args.turns,
        "temperature": args.temperature,
    }
    (out_dir / "synthesis.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
