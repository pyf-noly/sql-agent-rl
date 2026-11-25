#!/usr/bin/env python3
"""在租来的机器上一键取 BIRD 数据（dev 全量 + train 只解 RL 留出库）。

为什么需要它
------------
BIRD 官方给的是两个 zip，而且 train.zip 里还藏了**第二层** zip：

    train.zip (8.9GB)            → train/ (9.35GB)
      └── train_databases.zip (8.7GB) → train_databases/ (32.4GB)

全解压要 34GB 磁盘、耗时十几分钟；但 P6 实际只用到：

- **dev** 的 11 个库（评测，1.5GB）；
- **RL 留出库**（GRPO 的奖励要执行 SQL；由 `data/bird/db_split.json` 指定，约 8.4GB）。

SFT **不需要任何数据库**（prompt 里的 schema 已经在 jsonl 里了）。
所以这个脚本默认只解 RL 留出库，把磁盘从 34GB 压到 ~10GB。

用法::

    # 默认：dev 全量 + RL 留出库（需要先有 data/bird/db_split.json）
    python scripts/fetch_bird.py --out data/raw/bird

    # 没有切分文件 / 想全解压（34GB，确认磁盘够）
    python scripts/fetch_bird.py --out data/raw/bird --all

    # 已下过 zip，只想解压
    python scripts/fetch_bird.py --skip-download --out data/raw/bird
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URLS = {
    "dev.zip": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
    "train.zip": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
}


def human(n: float) -> str:
    return f"{n / 1e9:.2f}GB" if n >= 1e9 else f"{n / 1e6:.0f}MB"


def download(url: str, out: Path, skip: bool) -> None:
    if skip or out.is_file():
        if out.is_file():
            print(f"[skip] {out.name} 已存在（{human(out.stat().st_size)}）")
            return
        print(f"[skip] {out.name} 不存在但 --skip-download，跳过")
        return
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        got, t0 = 0, time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (500 << 20) < (1 << 20):
                    mb = got / max(1e-6, time.time() - t0) / 1e6
                    print(f"  {out.name} {human(got)}/{human(total)} ({mb:.1f}MB/s)", flush=True)
    os.replace(tmp, out)
    print(f"[ok] {out.name} {human(got)} in {time.time() - t0:.0f}s")


def extract_dev(bird: Path) -> None:
    dev_dir = bird / "dev_20240627"
    if (dev_dir / "dev.json").is_file():
        print(f"[skip] {dev_dir} 已解压")
        return
    z = bird / "dev.zip"
    print(f"[解压] {z.name} …")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(bird)
    inner = dev_dir / "dev_databases.zip"
    if inner.is_file():
        print("[解压] dev_databases.zip（1.5GB）…")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(dev_dir)
        inner.unlink()
    print(f"[ok] dev → {dev_dir}")


def extract_train(bird: Path, rl_dbs: list[str] | None) -> None:
    train_dir = bird / "train"
    z = bird / "train.zip"
    if not z.is_file() and not (train_dir / "train_databases.zip").is_file():
        print("[跳过] 没有 train.zip，也不做 train 解压")
        return
    if not (train_dir / "train.json").is_file():
        print(f"[解压] {z.name}（9.3GB）…")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(bird)
    inner = train_dir / "train_databases.zip"
    if not inner.is_file():
        print("[跳过] train_databases.zip 不存在（可能已解压过）")
        return
    dbs_dir = train_dir / "train_databases"
    dbs_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(inner) as zf:
        members = zf.namelist()
        # 实际布局是 ``train_databases/<db>/...``（1525 个成员，含 __MACOSX 噪音）
        if rl_dbs:
            want = set(rl_dbs)
            picked = [m for m in members
                      if not m.startswith("__MACOSX") and len(m.split("/")) > 1
                      and m.split("/")[1] in want]
            total = sum(zf.getinfo(m).file_size for m in picked)
            print(f"[解压] train_databases.zip 只取 {len(want)} 个 RL 库，"
                  f"{len(picked)} 个文件 {human(total)} …")
        else:
            picked = [m for m in members if not m.startswith("__MACOSX")]
            print("[解压] train_databases.zip 全量（约 32GB）…")
        for i, m in enumerate(picked, 1):
            zf.extract(m, train_dir)
            if i % 200 == 0:
                print(f"  {i}/{len(picked)}", flush=True)
    print(f"[ok] train dbs → {dbs_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/bird")
    ap.add_argument("--all", action="store_true", help="train 全解压（32GB）")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--split", default="data/bird/db_split.json",
                    help="含 rl_dbs 的切分文件；没有就退化成全解压")
    ap.add_argument("--keep-zips", action="store_true", help="解压后保留 zip（默认删，省 9GB）")
    args = ap.parse_args()

    bird = ROOT / args.out
    bird.mkdir(parents=True, exist_ok=True)

    split_p = ROOT / args.split
    rl_dbs = None
    if split_p.is_file():
        rl_dbs = json.loads(split_p.read_text(encoding="utf-8")).get("rl_dbs") or None
    if args.all:
        rl_dbs = None
    if rl_dbs is None:
        print("[提示] 没有 db_split.json 或指定了 --all：train 将全量解压（约 32GB）")
    else:
        print(f"[计划] RL 留出库 {len(rl_dbs)} 个：{rl_dbs[:6]}…")

    download(URLS["dev.zip"], bird / "dev.zip", args.skip_download)
    download(URLS["train.zip"], bird / "train.zip", args.skip_download)
    extract_dev(bird)
    extract_train(bird, rl_dbs)

    if not args.keep_zips:
        for z in (bird / "dev.zip", bird / "train.zip", bird / "train" / "train_databases.zip"):
            if z.is_file():
                z.unlink()
                print(f"[清理] 删除 {z.name}（--keep-zips 可保留）")

    used = sum(p.stat().st_size for p in bird.rglob("*") if p.is_file())
    print(f"\n[完成] {bird} 共 {human(used)}")
    print("下一步：python scripts/check_p6_env.py --model /root/autodl-tmp/qwen3-4b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
