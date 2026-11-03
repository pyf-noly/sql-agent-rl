#!/usr/bin/env python3
"""从 spider_data.zip 里只解出 dev/train 用到的数据库（20 个左右，约 20~60 MB 落盘）。

为什么不能只解一部分还要先有 zip
--------------------------------
zip 的最小可读单元是"中央目录 + 局部文件头"，没有随机读取协议就没法只拉一个库。
所以先把整包拿到本地（浏览器下最稳），再在本地做选择性解压。

用法::

    python scripts/extract_needed_dbs.py --zip "/mnt/e/下载/spider_data.zip"
    python scripts/extract_needed_dbs.py --zip /tmp/spider_data.zip --also-train

输出::

    data/raw/spider/database/<db_id>/<db_id>.sqlite
    data/raw/spider/tables.json          （顺手）
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ZIP_ROOT = "spider_data"


def _needed_db_ids(data_dir: Path, also_train: bool) -> set[str]:
    ids: set[str] = set()
    files = ["dev.json"] + (["train_spider.json"] if also_train else [])
    for f in files:
        p = data_dir / f
        if not p.is_file():
            print(f"[跳过] {f} 不存在")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        ids |= {r["db_id"] for r in rows}
        print(f"[读取] {f}: {len(rows)} 条，累计 {len(ids)} 个库")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="spider_data.zip 的路径")
    ap.add_argument("--out", default="data/raw/spider")
    ap.add_argument("--also-train", action="store_true", help="连 train 用到的库一起解")
    ap.add_argument("--all-dbs", action="store_true", help="解全部 200 个库")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = (root / args.out).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.zip).expanduser().resolve()
    if not zip_path.is_file():
        print(f"[失败] zip 不存在: {zip_path}")
        return 1
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[zip] {zip_path}  ({size_mb:.1f} MB)")
    if size_mb < 190:
        print(f"[警告] 完整包应为 196.3 MB，当前只有 {size_mb:.1f} MB —— 很可能没下完")

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        print(f"[失败] zip 损坏: {e}")
        print("      → 用浏览器重新下（勾选续传），或重下完整 196 MB")
        return 1

    with zf:
        names = zf.namelist()
        root_prefix = ZIP_ROOT + "/" if any(n.startswith(ZIP_ROOT + "/") for n in names) else ""
        print(f"[zip] 根目录: {root_prefix or '(无)'}，共 {len(names)} 个条目")

        wanted_ids = None
        if not args.all_dbs:
            wanted_ids = _needed_db_ids(data_dir, also_train=args.also_train)
            if not wanted_ids:
                print("[失败] 没有可用的 dev.json，先跑 download_spider.py --meta-only")
                return 1

        extracted = 0
        got_ids: set[str] = set()
        # tables.json
        for n in names:
            if n.endswith("/tables.json") or n == "tables.json":
                (data_dir / "tables.json").write_bytes(zf.read(n))
                print("[解压] tables.json")
                break

        for n in names:
            if n.endswith("/"):
                continue
            rel = n[len(root_prefix):] if root_prefix and n.startswith(root_prefix) else n
            parts = rel.split("/")
            if len(parts) != 3 or parts[0] != "database" or not parts[2].endswith(".sqlite"):
                continue
            db_id = parts[1]
            if wanted_ids is not None and db_id not in wanted_ids:
                continue
            target = data_dir / "database" / db_id / f"{db_id}.sqlite"
            if target.is_file() and target.stat().st_size > 0:
                got_ids.add(db_id)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
            got_ids.add(db_id)
            extracted += 1
            print(f"\r[解压] {len(got_ids)} 个库: {db_id}" + " " * 20, end="", flush=True)

    print()
    print(f"[结果] 落盘 {len(got_ids)} 个库，本次新解压 {extracted} 个")

    if wanted_ids is not None:
        missing = wanted_ids - got_ids
        if missing:
            print(f"[警告] {len(missing)} 个库没在 zip 里找到: {sorted(missing)[:5]}")
            return 1

    total = sum(p.stat().st_size for p in (data_dir / "database").rglob("*.sqlite"))
    print(f"[结果] database/ 总计 {total/1e6:.1f} MB")
    print("\n下一步：")
    print("  python - <<'PY'")
    print("from envs.db_registry import DatabaseRegistry")
    print('reg = DatabaseRegistry("data/raw/spider/database")')
    print("print(len(reg.list_db_ids()), reg.list_db_ids()[:3])")
    print("print(reg.schema(reg.list_db_ids()[0]).render_ddl()[:300])")
    print("PY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
