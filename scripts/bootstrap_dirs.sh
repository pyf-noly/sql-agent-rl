#!/usr/bin/env bash
# P0 Step 2：在仓库根目录创建全部目录骨架（可重复执行，已存在则跳过）。
# 用法： bash scripts/bootstrap_dirs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "仓库根目录: $ROOT"

dirs=(
  configs
  data/raw data/processed data/synthetic
  envs agents rewards rollout
  scripts analysis tests results runs docs
)
for d in "${dirs[@]}"; do
  mkdir -p "$d"
done

# 包标记：让 import envs.xxx 能用（Windows 上也能工作）
for d in envs agents rewards rollout scripts tests; do
  [ -f "$d/__init__.py" ] || : > "$d/__init__.py"
done

echo
echo "已创建目录（tree 视图）："
find . -maxdepth 2 -type d -not -path './.git*' | sort | sed 's|^\./||'

echo
echo "文件检查："
for f in .gitignore check_env.py .env.example; do
  if [ -f "$f" ]; then echo "  [OK]   $f"; else echo "  [MISS] $f"; fi
done

echo
echo "下一步："
echo "  1) cp .env.example .env  然后填入你的 API key"
echo "  2) python check_env.py"
