"""P0 环境自检：Python / 依赖 / 目录 / .gitignore / .env / SQLite / API 一次跑完。

用法（在仓库根目录）::

    conda activate sqlrl
    python check_env.py

全部通过返回 0，任意一项失败返回 1。
"""

from __future__ import annotations

import importlib
import os
import sqlite3
import sys

OK, FAIL = "[OK]  ", "[FAIL]"
results: list[tuple[bool, str]] = []


def _load_dotenv() -> None:
    """读 .env；dotenv 没装时降级为不读，让检查项自己报缺失依赖。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=True)


_load_dotenv()


def check(name: str, fn) -> None:
    try:
        results.append((True, f"{OK} {name}: {fn()}"))
    except Exception as e:  # noqa: BLE001 - 自检脚本要兜住一切异常
        results.append((False, f"{FAIL} {name}: {type(e).__name__}: {e}"))


def _python() -> str:
    assert sys.version_info >= (3, 10), f"需要 >=3.10，当前 {sys.version.split()[0]}"
    assert sys.version_info < (3, 13), (
        f"当前 {sys.version.split()[0]}，建议 3.10~3.12（P4 装 torch/verl 时 3.13+ 会踩坑）"
    )
    return sys.version.split()[0]


def _deps() -> str:
    need = ["openai", "dotenv", "httpx", "pytest", "sqlparse"]
    missing = []
    for m in need:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    assert not missing, f"缺少依赖: {missing}（pip install {' '.join(missing)}）"
    import openai
    return f"{len(need)} 个可用，openai=={getattr(openai, '__version__', '?')}"


def _dirs() -> str:
    need = [
        "configs", "data/raw", "data/processed", "data/synthetic",
        "envs", "agents", "rewards", "rollout",
        "scripts", "analysis", "tests", "results", "runs", "docs",
    ]
    missing = [d for d in need if not os.path.isdir(d)]
    assert not missing, f"缺少目录: {missing}（运行 bash scripts/bootstrap_dirs.sh 可自动创建）"
    return f"{len(need)} 个目录齐全"


def _gitignore() -> str:
    assert os.path.isfile(".gitignore"), ".gitignore 不存在"
    with open(".gitignore", encoding="utf-8") as f:
        text = f.read()
    for pat in [".env", "data/raw", "runs/", "__pycache__"]:
        assert pat in text, f".gitignore 缺少 {pat}"
    return ".env / data/raw / runs / __pycache__ 均已忽略"


def _env_vars() -> str:
    for k in ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"]:
        assert os.getenv(k), f".env 缺少 {k}"
    url = os.getenv("OPENAI_BASE_URL", "")
    assert url.startswith("http"), f"OPENAI_BASE_URL 不是合法 URL: {url}"
    return f"{url} / model={os.getenv('OPENAI_MODEL')} / key=***{os.getenv('OPENAI_API_KEY', '')[-4:]}"


def _sqlite() -> str:
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE t(a INT)")
        con.execute("INSERT INTO t VALUES (1)")
        assert con.execute("SELECT a FROM t").fetchone()[0] == 1
        # 顺手验证只读 URI 能打开（P1 的沙箱依赖它）
        con.close()
        con = sqlite3.connect("file::memory:?cache=shared", uri=True)
        assert con.execute("SELECT 1").fetchone()[0] == 1
    finally:
        con.close()
    return f"sqlite3 {sqlite3.sqlite_version}，只读 URI 可用"


def _api() -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=30.0,
    )
    r = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[{"role": "user", "content": "只回答两个字：收到"}],
        temperature=0,
        max_tokens=8,
    )
    text = (r.choices[0].message.content or "").strip()
    assert text, "接口返回空内容"
    return f"生成成功，返回 {text!r}"


check("Python 版本", _python)
check("依赖包", _deps)
check("目录结构", _dirs)
check(".gitignore", _gitignore)
check(".env 三件套", _env_vars)
check("SQLite", _sqlite)
check("API 连通", _api)

print("\n" + "=" * 64)
for _, line in results:
    print(line)
print("=" * 64)
bad = sum(1 for ok, _ in results if not ok)
print(f"\n结果: {len(results) - bad}/{len(results)} 通过\n")
sys.exit(1 if bad else 0)
