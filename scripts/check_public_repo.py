#!/usr/bin/env python3
"""上传 GitHub 之前的自检：大文件、密钥、不该进库的目录。

为什么需要
----------
GitHub 的两条硬限制会让 push 直接失败，而且失败信息很不友好：

- **单文件 >100MB**：push 被拒；>50MB 会警告。本项目原始数据 42GB，必须拦住；
- **密钥泄漏**：一旦 push 上去，就算删了文件，历史里还在（要 rewrite history）。

用法::

    python scripts/check_public_repo.py            # 检查仓库
    python scripts/check_public_repo.py --max-mb 50
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 不扫描的目录（生成物/缓存/数据集）
SKIP_DIRS = {
    ".git", ".cache", ".pytest_cache", ".pytest-run", ".pytest_tmp",
    "__pycache__", "dist", ".venv", "venv", ".tools", ".models",
    "data/raw", "runs",
}
#: 明确允许出现的“像密钥”的文件
ALLOW_FILES = {".env.example", "check_public_repo.py", "GITHUB_UPLOAD.md"}

#: 只在本地、已被 .gitignore 排除的敏感文件（发现时只警告，不算失败）
LOCAL_ONLY = re.compile(r"^(\.env|\.env\..*|.*\.key|secrets/)")


def is_local_only(rel: str) -> bool:
    return bool(LOCAL_ONLY.match(rel)) and rel != ".env.example"

SECRET_PATTERNS = [
    ("OpenAI/DeepSeek 风格 key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("私钥", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("硬编码密码/密钥", re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*"
        r"['\"][^'\"\s]{8,}['\"]")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
]


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        yield p, rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=50.0, help="单文件上限（MB）")
    args = ap.parse_args()

    big: list[tuple[str, float]] = []
    secrets: list[tuple[str, int, str, str]] = []
    local_keys: list[str] = []
    total = 0
    n = 0

    for p, rel in iter_files():
        n += 1
        size = p.stat().st_size
        total += size
        if size > args.max_mb * 1024 * 1024:
            big.append((rel, size / 1024 / 1024))
        if is_local_only(rel):
            local_keys.append(rel)
            continue
        if size > 2 * 1024 * 1024 or p.suffix.lower() in {".png", ".jpg", ".jpeg", ".zip", ".sqlite"}:
            continue
        if p.name in ALLOW_FILES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, pat in SECRET_PATTERNS:
                if pat.search(line):
                    secrets.append((rel, i, name, line.strip()[:80]))

    print(f"扫描 {n} 个文件，合计 {total/1024/1024:.1f} MB（含本机忽略文件）")
    ok = True

    if big:
        ok = False
        print(f"\n[失败] {len(big)} 个文件超过 {args.max_mb:g}MB（GitHub 单文件上限 100MB）：")
        for rel, mb in sorted(big, key=lambda x: -x[1])[:10]:
            print(f"  {mb:8.1f} MB  {rel}")
    else:
        print(f"[通过] 没有文件超过 {args.max_mb:g}MB")

    if secrets:
        ok = False
        print(f"\n[失败] 发现 {len(secrets)} 处疑似密钥：")
        for rel, line, name, snippet in secrets[:10]:
            print(f"  {rel}:{line}  ({name})  {snippet}")
    else:
        print("[通过] 没有发现疑似密钥")

    if local_keys:
        print(f"\n[注意] 本机有 {len(local_keys)} 个敏感文件，已被 .gitignore 排除、不会上传：")
        for rel in local_keys[:5]:
            print(f"  {rel}")
        print("       如果这些 key 出现在截图/日志/聊天记录里，建议去控制台轮换一次。")

    print("\n" + ("[结论] 可以上传" if ok else "[结论] 先处理上面的问题再上传"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
