#!/usr/bin/env python3
"""把若干 results/baseline_*.json 汇总成一张 Markdown 表（写报告/简历直接粘）。

为什么需要
----------
P5 结束时你会有 5~6 个结果文件（DeepSeek 的两个、SFT 的两个、GRPO 的两个），
手动抄进表格极易错，而且"分难度 + 执行失败 + 差值"每次都要重算。
这个脚本只做一件事：**从 JSON 里把数字取出来，排成一张能直接贴进 README 的表**。

用法::

    # 指定文件，按给定顺序出现在表里
    python scripts/compare_results.py \
        results/baseline_sft_t1.json results/baseline_sft_t3.json \
        results/baseline_p5_t1.json results/baseline_p5_t3.json

    # 或者用通配符（按文件名排序）
    python scripts/compare_results.py "results/baseline_*.json"

    # 加 --csv 输出 CSV
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIFFS = ["easy", "medium", "hard", "extra"]


def _tag(p: Path, d: dict) -> str:
    cfg = d.get("config") or {}
    if cfg.get("tag"):
        return str(cfg["tag"])
    return p.stem.replace("baseline_", "")


def _row(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("metrics") or {}
    by = m.get("by_difficulty") or {}
    cfg = d.get("config") or {}
    return {
        "file": p.name,
        "tag": _tag(p, d),
        "n": m.get("total", 0),
        "ea": m.get("accuracy", 0.0),
        "correct": m.get("correct", 0),
        "easy": (by.get("easy") or {}).get("accuracy"),
        "medium": (by.get("medium") or {}).get("accuracy"),
        "hard": (by.get("hard") or {}).get("accuracy"),
        "extra": (by.get("extra") or {}).get("accuracy"),
        "pred_error": m.get("pred_error", 0),
        "turns": cfg.get("max_turns"),
        "until_correct": cfg.get("until_correct"),
        "model": cfg.get("model"),
        "elapsed_s": cfg.get("elapsed_sec"),
    }


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _render(rows: list[dict], markdown: bool) -> str:
    head = ["设置", "样本", "EA", "easy", "medium", "hard", "extra", "执行失败"]
    body = []
    for r in rows:
        label = r["tag"]
        if r["turns"] and r["until_correct"]:
            label += f"（{r['turns']} 轮·判对停）"
        elif r["turns"] and r["turns"] > 1:
            label += f"（{r['turns']} 轮）"
        body.append([label, str(r["n"]), _pct(r["ea"]), _pct(r["easy"]),
                     _pct(r["medium"]), _pct(r["hard"]), _pct(r["extra"]),
                     str(r["pred_error"])])

    # 相邻两行差值（只有一个基准时也给出"最后一行 - 第一行"）
    deltas = []
    for a, b in zip(rows, rows[1:]):
        if a["n"] == b["n"]:
            d = (b["ea"] - a["ea"]) * 100
            deltas.append((b["tag"], a["tag"], d))

    if not markdown:
        widths = [max(len(str(x[i])) for x in [head] + body) for i in range(len(head))]
        lines = ["  ".join(str(c).ljust(w) for c, w in zip(head, widths)),
                 "  ".join("-" * w for w in widths)]
        lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in body]
        return "\n".join(lines)

    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(row) + " |" for row in body]
    if deltas:
        out.append("")
        out.append("差值：" + "；".join(f"{t} − {f} = **{d:+.2f}pp**" for t, f, d in deltas))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="results/baseline_*.json，支持通配符")
    ap.add_argument("--markdown", action="store_true", default=True)
    ap.add_argument("--plain", action="store_true", help="对齐的纯文本表")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    paths: list[Path] = []
    for f in args.files:
        for hit in glob.glob(f) or ([f] if Path(f).is_file() else []):
            p = Path(hit)
            if not p.is_absolute():
                p = ROOT / p
            if p.is_file() and p not in paths:
                paths.append(p)

    if not paths:
        print("[失败] 没有找到任何结果文件", file=sys.stderr)
        return 1

    rows = [_row(p) for p in paths]
    rows = [r for r in rows if r["n"] > 0]

    if args.csv:
        cols = ["tag", "n", "ea", "easy", "medium", "hard", "extra", "pred_error", "turns"]
        print(",".join(cols))
        for r in rows:
            print(",".join(str(r[c]) for c in cols))
        return 0

    print(f"\n共 {len(rows)} 个结果\n")
    print(_render(rows, markdown=not args.plain))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
