#!/usr/bin/env python3
"""检查 .env 里的 API key 是否可用（只打印打码后的 key）。

用法（在仓库根目录）::

    python scripts/check_api_key.py

输出三种结果：
- ✅ key 可用
- ❌ 401：key 无效/已删除/复制不全
- ❌ 402：key 有效但余额不足
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[错误] 缺 python-dotenv：pip install python-dotenv")
        return 2
    load_dotenv(ROOT / ".env", override=True)  # 以 .env 文件为准，避免系统环境变量盖住

    base = os.environ.get("OPENAI_BASE_URL", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    if not base or not key:
        print("[错误] .env 里缺 OPENAI_BASE_URL 或 OPENAI_API_KEY")
        return 2

    masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else "(太短，可能是占位符)"
    print(f"base_url = {base}")
    print(f"model    = {model}")
    print(f"key      = {masked}")

    try:
        from openai import OpenAI
    except ImportError:
        print("[错误] 缺 openai：pip install openai")
        return 2

    try:
        client = OpenAI(base_url=base, api_key=key, timeout=30.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复 OK 两个字母"}],
            max_tokens=8,
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"[通过] key 可用，模型回复：{text}")
        return 0
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"[失败] {type(e).__name__}: {msg[:300]}")
        if "401" in msg or "Authentication" in msg:
            print("→ key 无效、已删除，或复制时少了字符")
        elif "402" in msg or "Insufficient Balance" in msg:
            print("→ key 有效，但账户余额不足")
        elif "Connection" in msg or "timeout" in msg.lower():
            print("→ 网络问题：检查代理/网络后重试")
        return 1


if __name__ == "__main__":
    sys.exit(main())
