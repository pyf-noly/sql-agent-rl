#!/usr/bin/env python3
"""从 ModelScope 直接下载 Qwen3-4B-Instruct-2507（不装 modelscope 库）。

为什么不在租卡机上用 ``huggingface-cli``
--------------------------------------
国内机器连 HF 经常超时/被限速；P4 的坑表里已经记过一条（hf_transfer + 代理 → 直接失败）。
ModelScope 的 HTTP 接口稳定、可断点续传，而且**没有额外依赖**——只用标准库。

用法::

    python scripts/download_model_ms.py --out /root/autodl-tmp/qwen3-4b
    python scripts/download_model_ms.py --out ./qwen3-4b --repo Qwen/Qwen3-4B-Instruct-2507

断点续传：文件先写 ``*.part``，完成后原子改名；重复执行只会补没下完的。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request

API_FILES = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"
API_RAW = ("https://www.modelscope.cn/api/v1/models/{repo}/repo"
           "?Revision=master&FilePath={path}")

#: 只下跑模型必需的文件（README/LICENSE 之类不要，省时间）
WANT_SUFFIX = (".json", ".txt", ".safetensors")


def list_files(repo: str) -> list[str]:
    url = API_FILES.format(repo=repo)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    import json

    data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    files = [f["Name"] for f in data["Data"]["Files"]]
    return [f for f in files if f.endswith(WANT_SUFFIX)]


def download(repo: str, name: str, dst_dir: str) -> int:
    out = os.path.join(dst_dir, name)
    url = API_RAW.format(repo=repo, path=urllib.parse.quote(name))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if os.path.exists(out) and total and os.path.getsize(out) == total:
            print(f"[skip] {name} {total / 1e6:.1f}MB", flush=True)
            return 0
        tmp = out + ".part"
        got = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (500 << 20) < (1 << 20):
                    speed = got / max(1e-6, time.time() - t0) / 1e6
                    print(f"  {name} {got / 1e6:.0f}/{(total or got) / 1e6:.0f}MB "
                          f"({speed:.1f}MB/s)", flush=True)
    os.replace(tmp, out)
    print(f"[ok] {name} {got / 1e6:.1f}MB in {time.time() - t0:.0f}s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="/root/autodl-tmp/qwen3-4b")
    ap.add_argument("--only-weights", action="store_true",
                    help="只下 safetensors（配置/tokenizer 已经有了时用）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    try:
        files = list_files(args.repo)
    except Exception as e:  # noqa: BLE001
        print(f"[失败] 取文件列表失败: {type(e).__name__}: {e}")
        print("       检查网络；或改用 modelscope 的 snapshot_download。")
        return 1
    if args.only_weights:
        files = [f for f in files if f.endswith(".safetensors")]
    # 小文件先下（配置/tokenizer），权重最后——中途断掉也能立刻做 tokenizer 检查
    files.sort(key=lambda f: (f.endswith(".safetensors"), f))
    print(f"[计划] {args.repo} → {args.out}，{len(files)} 个文件")
    for f in files:
        download(args.repo, f, args.out)
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in files)
    print(f"[完成] {len(files)} 个文件，合计 {total / 1e9:.2f}GB → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
