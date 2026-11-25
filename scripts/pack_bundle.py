#!/usr/bin/env python3
"""把"上卡要用的东西"打成一个 zip，避免传一堆没用的缓存。

包含：代码 + 配置 + data/processed（几 MB）+ 文档；
不包含：.git / .cache / runs / results / __pycache__ / 测试临时目录。

数据库（约 490MB）默认**不打包**（--with-db 才带）：
Spider 的 sqlite 文件压缩率一般，传之前先确认你的上传通道够快；
如果走 AutoDL 的"阿里云盘"通道，带不带都行。

用法::

    python scripts/pack_bundle.py                    # 只打包代码与数据
    python scripts/pack_bundle.py --with-db          # 连 sqlite 一起（约 250~300MB）
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git", ".cache", ".pytest_cache", ".venv", "venv", "__pycache__",
    "runs", "results", "dist", ".idea", ".vscode",
    "pytest-cache-files-nawl6tq2", "pytest-cache-files-w4bafm1h",
    "p1check-0k3a3xt2", "_p1verify", ".pytest_tmp", ".pytest-run",
}
SKIP_SUFFIX = {".pyc", ".pyo"}
#: 体积大、可由脚本重新生成的东西。
#: 注意：``data/processed/sft_final.jsonl`` **必须带上**——``build_p4_data.py``
#: 的输入就是它，漏了就得在租来的机器上重跑 P3（要花钱调 API）。
SKIP_FILES = {"data/processed/sft_coldstart.jsonl", "data/processed/sft_seeds.json",
              "data/synthetic/episodes.jsonl.bak"}
#: 数据库只在机器上下载，绝不进包：Spider 490MB 已经不划算，
#: BIRD 的 train_databases 解压后是 25GB，打包会把上传通道直接堵死。
SKIP_DB_PREFIXES = ("data/raw/spider/database", "data/raw/bird/")


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/train_bundle.zip")
    ap.add_argument("--with-db", action="store_true", help="（仅 Spider）把 sqlite 也打进去")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    n, total = 0, 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = _rel(p)
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.is_dir() or p.suffix in SKIP_SUFFIX:
                continue
            if rel in SKIP_FILES or p.name == ".env":
                continue
            if rel.startswith("data/raw/bird/") or out == p:
                continue
            if rel.startswith("data/raw/spider/database") and not args.with_db:
                continue
            z.write(p, rel)
            n += 1
            total += p.stat().st_size

    print(f"[打包] {out.relative_to(ROOT)}")
    print(f"[内容] {n} 个文件，原始 {total / 2**20:.0f}MB，压缩后 {out.stat().st_size / 2**20:.0f}MB")
    if not args.with_db:
        print("[提醒] 未包含任何数据库（Spider / BIRD）：到机器上单独下载。")
        print("       Spider: data/raw/spider/database/   BIRD: data/raw/bird/{train,dev}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
