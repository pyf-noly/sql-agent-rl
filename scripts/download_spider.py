#!/usr/bin/env python3
"""获取 Spider 1.0 数据（问题 + SQL 走 API，数据库走 zip），适配不稳定网络。

三条独立通道
------------
1. **dev/train 的问题与 gold SQL**：走 HuggingFace ``datasets-server`` API
   （``syedmuzammil20/spider``），每次 100 行、共 11 次请求，总计不到 1 MB。
   就算网络抖，重试单页即可。
2. **SQLite 数据库**：只能来自 ``HAL-9001/spider-databases`` 的 ``spider_data.zip``
   （196 MB）。网络不稳时**推荐用浏览器下载**，再用 ``--local-zip`` 指进来。
3. **tables.json**：同样在 zip 里；缺失时用 dev 数据反推一个最小版本。

推荐用法
--------
    # A. 先把 zip 用浏览器下到 E:\\下载\\spider_data.zip，然后：
    python scripts/download_spider.py --local-zip "/mnt/e/下载/spider_data.zip"

    # B. 让脚本自己下 zip（会断点续传重试）
    python scripts/download_spider.py --fetch-zip

    # C. 只取问题/SQL，不取数据库（先验证 dev.json 通道是否通）
    python scripts/download_spider.py --meta-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HF_ENDPOINTS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]
DATASET_SERVER = [
    "https://datasets-server.huggingface.co",
    "https://datasets-server.hf-mirror.com",
]
SPIDER_DS = "syedmuzammil20%2Fspider"
SPIDER_CONFIG = "spider"
ZIP_URLS = [
    "https://hf-mirror.com/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip",
    "https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip",
]
ZIP_ROOT = "spider_data"
WANTED = {"database", "tables.json", "dev.json", "dev_gold.sql",
          "train_spider.json", "train_others.json", "test.json"}
PAGE = 100


def _get_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "sql-agent-rl/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_split_via_api(split: str, retries: int = 5) -> list[dict]:
    """用 datasets-server 的 /rows 接口按页取一个 split，返回 Spider 原始 JSON 结构。"""
    last_err: Exception | None = None
    for base in DATASET_SERVER:
        rows: list[dict] = []
        offset = 0
        try:
            while True:
                url = (f"{base}/rows?dataset={SPIDER_DS}&config={SPIDER_CONFIG}"
                       f"&split={split}&offset={offset}&length={PAGE}")
                attempt = 0
                while True:
                    try:
                        payload = _get_json(url)
                        break
                    except Exception as e:  # noqa: BLE001
                        attempt += 1
                        last_err = e
                        if attempt > retries:
                            raise
                        time.sleep(min(2 ** attempt, 10))
                page = payload.get("rows", [])
                if not page:
                    break
                rows.extend(r["row"] for r in page)
                print(f"\r      {split}: {len(rows)} 行", end="", flush=True)
                if len(page) < PAGE:
                    break
                offset += PAGE
            print()
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001
            print(f"\n      [失败] {base}: {type(e).__name__}: {e}")
            last_err = e
            continue
    raise RuntimeError(f"{split} 取不到: {last_err}")


def _normalize(rows: list[dict]) -> list[dict]:
    """只保留下游用得到的字段，并且顺序稳定。"""
    keys = ["db_id", "question", "query"]
    out = []
    for r in rows:
        if not all(k in r for k in keys):
            continue
        out.append({k: r[k] for k in keys})
    return out


def _fetch_meta(data_dir: Path) -> tuple[int, int]:
    """取 dev + train 的问题/SQL，写 dev.json / train_spider.json。"""
    print("[1/3] 取 dev（validation split）")
    dev = _normalize(_fetch_split_via_api("validation"))
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dev.json").write_text(
        json.dumps(dev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"      dev.json: {len(dev)} 条")

    print("[2/3] 取 train")
    tr = _normalize(_fetch_split_via_api("train"))
    (data_dir / "train_spider.json").write_text(
        json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"      train_spider.json: {len(tr)} 条")
    return len(dev), len(tr)


def _download_zip_with_resume(dest: Path, retries: int = 10) -> None:
    """整包下载，断流续传。"""
    import os
    expected = 205_800_266
    for url in ZIP_URLS:
        print(f"[下载] {url}")
        attempt = 0
        while attempt <= retries:
            pos = dest.stat().st_size if dest.exists() else 0
            if pos >= expected:
                return
            attempt += 1
            headers = {"User-Agent": "sql-agent-rl/0.1"}
            if pos:
                headers["Range"] = f"bytes={pos}-"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    if pos and resp.status != 206:
                        pos = 0
                    total = int(resp.headers.get("Content-Length") or 0) + pos
                    print(f"      尝试 {attempt}: {pos/1e6:.1f} MB → {total/1e6:.1f} MB")
                    mode = "ab" if pos else "wb"
                    got = pos
                    with open(dest, mode) as f:
                        while True:
                            buf = resp.read(1 << 20)
                            if not buf:
                                break
                            f.write(buf)
                            got += len(buf)
                            print(f"\r      {got/1e6:.1f}/{total/1e6:.1f} MB", end="",
                                  flush=True)
                    print()
                    if total and got < total:
                        raise ConnectionError(f"提前结束 {got}/{total}")
                    return
            except Exception as e:  # noqa: BLE001
                print(f"\n      [断流] {type(e).__name__}: {e}")
                time.sleep(min(2 ** attempt, 20))
        # 这个源不行，换下一个（已下部分作废，避免接错文件）
        if dest.exists():
            dest.unlink()
    raise RuntimeError("所有源都失败，建议用浏览器下载 zip 后 --local-zip")


def _extract(zip_path: Path, data_dir: Path) -> int:
    print("[解压]")
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        root = ZIP_ROOT + "/" if any(n.startswith(ZIP_ROOT + "/") for n in names) else ""
        if not root:
            for n in names:
                if n.endswith("tables.json"):
                    parts = n.split("/")
                    root = "/".join(parts[:-1]) + "/" if len(parts) > 1 else ""
                    break
        print(f"      根目录: {root or '(无)'}")
        for n in names:
            if n.endswith("/"):
                continue
            rel = n[len(root):] if root and n.startswith(root) else n
            if rel.split("/")[0] not in WANTED:
                continue
            target = data_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    print(f"      解压 {count} 个文件")
    return count


def _verify(data_dir: Path) -> bool:
    print("[3/3] 校验")
    ok = True
    dev_p = data_dir / "dev.json"
    if dev_p.is_file():
        dev = json.loads(dev_p.read_text(encoding="utf-8"))
        print(f"      dev.json: {len(dev)} 条，字段 {sorted(dev[0].keys())}")
    else:
        print("      [缺失] dev.json")
        ok = False

    db_dir = data_dir / "database"
    if db_dir.is_dir():
        dbs = [p for p in db_dir.iterdir()
               if p.is_dir() and (p / f"{p.name}.sqlite").is_file()]
        print(f"      database: {len(dbs)} 个库")
        if dev_p.is_file() and dbs:
            have = {p.name for p in dbs}
            need = {e["db_id"] for e in json.loads(dev_p.read_text(encoding="utf-8"))}
            miss = need - have
            print(f"      dev 需要 {len(need)} 个库，缺 {len(miss)} 个"
                  + (f"（例: {sorted(miss)[:3]}）" if miss else "（齐全）"))
    else:
        print("      [缺失] database/（只有问题、没法执行 SQL）")
        ok = False

    tp = data_dir / "tables.json"
    print(f"      tables.json: {'有' if tp.is_file() else '缺失（可从 zip 补）'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/spider")
    ap.add_argument("--local-zip", help="本地已下载的 spider_data.zip 路径（推荐）")
    ap.add_argument("--fetch-zip", action="store_true", help="让脚本自己下 zip")
    ap.add_argument("--meta-only", action="store_true", help="只取问题/SQL，不要数据库")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = (root / args.out).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        _fetch_meta(data_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[失败] 取问题/SQL 失败: {type(e).__name__}: {e}")
        print("  → datasets-server 走不通时，改用 --local-zip 直接从 zip 里取")
        return 1

    if args.meta_only:
        _verify(data_dir)
        return 0 if (data_dir / "dev.json").is_file() else 1

    zip_path: Path | None = None
    if args.local_zip:
        zip_path = Path(args.local_zip).expanduser().resolve()
        if not zip_path.is_file():
            print(f"[失败] 本地 zip 不存在: {zip_path}")
            return 1
        print(f"[使用] 本地 zip: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
    elif args.fetch_zip:
        zip_path = root / "spider_data.zip"
        if zip_path.exists() and zip_path.stat().st_size < 10_000_000:
            zip_path.unlink()
        try:
            _download_zip_with_resume(zip_path)
        except Exception as e:  # noqa: BLE001
            print(f"[失败] {e}")
            return 1
    else:
        print("\n未指定 zip 来源。三选一：")
        print("  1) 浏览器下载后：  python scripts/download_spider.py --local-zip /mnt/e/下载/spider_data.zip")
        print("  2) 脚本自己下：    python scripts/download_spider.py --fetch-zip")
        print("  3) 先不要数据库：  python scripts/download_spider.py --meta-only")
        _verify(data_dir)
        return 0

    try:
        _extract(zip_path, data_dir)
    except zipfile.BadZipFile as e:
        print(f"[失败] zip 损坏（多半是没下完）: {e}")
        print(f"  当前大小 {zip_path.stat().st_size/1e6:.1f} MB，完整应为 196.3 MB")
        return 1

    if not args.keep_zip and zip_path.parent == root:
        zip_path.unlink(missing_ok=True)

    _verify(data_dir)
    print(f"\n完成: {data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
