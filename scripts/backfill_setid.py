#!/usr/bin/env python3
"""给 episodes 回填/纠正 set_id，并按当前种子集重映射 idx（整表重建）。

为什么需要"整表重建"
--------------------
``synthesize_sft.py`` 早期只用全局 ``idx`` 做断点续跑键，后来加了
``--only-difficulty`` 就撞车了。修复过程中又加了"种子去重"，导致指纹变化，
旧记录带着过期指纹，用"只补缺失 set_id"的逻辑救不回来。

所以这里不猜：把整份 ``episodes.jsonl`` 读进来，按当前种子集重新对齐：

1. 复现当前的筛选 + 去重规则，得到种子列表与指纹；
2. 用 ``(db_id, question)`` 反查每条记录在种子列表里的位置；
3. 映射不上的（别的批次、dry-run 残留）按 ``--drop-unmapped`` 处理；
4. 同一 ``(db_id, question)`` 有多条时只留一条（优先留已判对的）；
5. 全量重写，先备份。

用法::

    python scripts/backfill_setid.py --only-difficulty hard,extra --drop-unmapped
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def seed_fingerprint(seeds: list[dict]) -> str:
    payload = json.dumps(
        [(s.get("db_id"), s.get("question"), s.get("difficulty")) for s in seeds],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/processed/sft_seeds.json")
    ap.add_argument("--episodes", default="data/synthetic/episodes.jsonl")
    ap.add_argument("--only-difficulty", default="hard,extra")
    ap.add_argument("--drop-unmapped", action="store_true",
                    help="丢掉映射不上的记录（别的批次 / dry-run 残留）")
    args = ap.parse_args()

    seeds_path = ROOT / args.seeds
    ep_path = ROOT / args.episodes
    if not seeds_path.is_file() or not ep_path.is_file():
        print(f"[失败] 缺文件: {seeds_path} / {ep_path}")
        return 1

    seeds_all = json.loads(seeds_path.read_text(encoding="utf-8"))
    want = {d.strip() for d in args.only_difficulty.split(",") if d.strip()}
    filtered = [s for s in seeds_all if s.get("difficulty") in want]

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for s in filtered:
        k = (s.get("db_id"), s.get("question"))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(s)
    if len(deduped) != len(filtered):
        print(f"[去重] {len(filtered)} -> {len(deduped)}")
    filtered = deduped

    fp = seed_fingerprint(filtered)
    print(f"[种子集] {len(filtered)} 条（筛选 {sorted(want)}），指纹 {fp}")

    pos_of = {(s["db_id"], s["question"]): i for i, s in enumerate(filtered)}

    records: list[dict] = []
    for line in ep_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    print(f"[读取] {len(records)} 条记录")

    # 按位置归并：同一位置只留一条，优先"判对"的，其次轮数多的
    best: dict[int, dict] = {}
    unmapped = 0
    for d in records:
        pos = pos_of.get((d.get("db_id"), d.get("question")))
        if pos is None:
            unmapped += 1
            continue
        cur = best.get(pos)
        if cur is None:
            best[pos] = d
            continue
        cur_ok = any(t.get("correct") for t in cur.get("turns") or [])
        new_ok = any(t.get("correct") for t in d.get("turns") or [])
        if new_ok and not cur_ok:
            best[pos] = d
        elif new_ok == cur_ok and len(d.get("turns") or []) > len(cur.get("turns") or []):
            best[pos] = d

    out_lines: list[str] = []
    for pos, d in sorted(best.items()):
        d["set_id"] = fp
        d["idx"] = pos
        d["difficulty"] = filtered[pos].get("difficulty")
        out_lines.append(json.dumps(d, ensure_ascii=False))

    backup = ep_path.with_suffix(ep_path.suffix + ".bak")
    shutil.copy2(ep_path, backup)
    ep_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[回填] {len(out_lines)} 条 -> set_id={fp}")
    if unmapped:
        action = "已丢弃" if args.drop_unmapped else "未写入（无法映射）"
        print(f"[注意] {unmapped} 条映射不上本批种子，{action}")
    print(f"[备份] {backup.relative_to(ROOT)}")

    missing = [i for i in range(len(filtered)) if i not in best]
    print(f"[校验] 已完成 {len(best)} / {len(filtered)}")
    print(f"[校验] 还缺 {len(missing)} 条" + (f"，前 5: {missing[:5]}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
