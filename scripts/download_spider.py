#!/usr/bin/env python3
"""下载 Spider 1.0 数据集（P1 可选步骤：真实数据用于 P2 评测）。

优先走 HuggingFace（国内可访问性较好，且有镜像），失败后依次回退。
下载体积约 100~400 MB，取决于镜像。

用法::

    cd "/mnt/e/deepseek工作区/sql-agent-rl"
    conda activate sqlrl
    python scripts/download_spider.py

成功后目录结构::

    data/raw/spider/
    ├── database/<db_id>/<db_id>.sqlite
    ├── train_spider.json
    ├── dev.json
    └── tables.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# 依次尝试的镜像。每个条目：(名称, URL, 解压后根目录特征文件)
MIRRORS: list[tuple[str, str]] = [
    ("hf-mirror", "https://hf-mirror.com/datasets/syedmuzammil20/spider/resolve/main/spider.zip"),
    ("huggingface", "https://huggingface.co/datasets/syedmuzammil20/spider/resolve/main/spider.zip"),
]

# 只要这些文件/目录，别把整个 zip 都铺开
WANTED = {"database", "tables.json", "dev.json", "dev_gold.sql",
          "train_spider.json", "train_others.json", "test.json"}

EXPECTED_FILES = ["tables.json", "dev.json", "train_spider.json"]


def _download(url: str, dest: Path) -> None:
    print(f"[1/3] 下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "sql-agent-rl/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        chunk = 1 << 20
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total:
                    pct = 100.0 * got / total
                    print(f"\r      {got/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()
    if dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"下载文件过小（{dest.stat().st_size} 字节），可能不是有效压缩包")


def _extract(zip_path: Path, dest_dir: Path) -> None:
    print("[2/3] 解压")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # 找到包含 tables.json 的那一层作为根
        roots = {n.split("/")[0] for n in names if "/" in n}
        if not roots:
            raise RuntimeError("压缩包结构异常：没有顶层目录")
        root = sorted(roots)[0]
        print(f"      压缩包根目录: {root}/")
        extracted = 0
        for n in names:
            rel = n[len(root) + 1:] if n.startswith(root + "/") else None
            if not rel:
                continue
            top = rel.split("/")[0]
            if top not in WANTED:
                continue
            target = dest_dir / rel
            if n.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
        print(f"      解压 {extracted} 个文件")


def _verify(data_dir: Path) -> None:
    print("[3/3] 校验")
    missing = [f for f in EXPECTED_FILES if not (data_dir / f).is_file()]
    if missing:
        raise RuntimeError(f"缺少文件: {missing}")
    db_dir = data_dir / "database"
    if not db_dir.is_dir():
        raise RuntimeError("缺少 database/ 目录")
    dbs = [p for p in db_dir.iterdir() if p.is_dir() and (p / f"{p.name}.sqlite").is_file()]
    print(f"      数据库数: {len(dbs)}")
    with open(data_dir / "dev.json", encoding="utf-8") as f:
        dev = json.load(f)
    print(f"      dev 样本数: {len(dev)}")
    print(f"      样例 db_id: {dev[0].get('db_id')}")
    print(f"      样例问题: {dev[0].get('question')[:60]}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/spider", help="输出目录（默认 data/raw/spider）")
    ap.add_argument("--keep-zip", action="store_true", help="保留下载的 zip")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = (root / args.out).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if all((data_dir / f).is_file() for f in EXPECTED_FILES):
        print(f"数据已存在: {data_dir}（跳过下载）")
        _verify(data_dir)
        return 0

    errors: list[str] = []
    for name, url in MIRRORS:
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "spider.zip"
            try:
                print(f"=== 尝试镜像: {name} ===")
                _download(url, zip_path)
                _extract(zip_path, data_dir)
                _verify(data_dir)
                if args.keep_zip:
                    shutil.copy2(zip_path, root / "spider.zip")
                print(f"\n完成: {data_dir}")
                return 0
            except Exception as e:  # noqa: BLE001
                print(f"[失败] {name}: {type(e).__name__}: {e}")
                errors.append(f"{name}: {e}")

    print("\n所有镜像都失败。请手动下载后放到 data/raw/spider/：")
    print("  官方页面: https://yale-lily.github.io/spider")
    print("  或 HuggingFace: https://huggingface.co/datasets/syedmuzammil20/spider")
    print("\n错误汇总:")
    for e in errors:
        print(" -", e)
    return 1


if __name__ == "__main__":
    sys.exit(main())
