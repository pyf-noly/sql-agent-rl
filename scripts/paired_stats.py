#!/usr/bin/env python3
"""对两个评测结果做**配对**比较（同一批题），不需要 GPU、不重新生成。

为什么需要它
-----------
评测结果里的 ``accuracy`` 只能做独立样本比较，SE 偏大；但本项目所有对照
（SFT vs GRPO、v1 vs v3、beta=0 vs beta=1）都是**同一批 307 条题**上跑的——
应该用配对比较：只看"一个对、一个错"的不一致样本（McNemar）。

    # 基础：两个文件的 EA 与差值
    python scripts/paired_stats.py results/a.json results/b.json

    # 批量：把某组结果两两比较（第一个是基线）
    python scripts/paired_stats.py results/baseline_bird_p6v3_t3_s5.json \
        results/baseline_bird_p6sft_t3_s5.json

输出：A/B 的 EA、差值、不一致对 (b, c)、配对 SE、z、精确 McNemar p、95% CI，
以及"要显著需要多少条 / 80% 功效需要多少条"的估算。

约定：``b`` = 只有 A 对、``c`` = 只有 B 对（diff = A − B）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def exact_mcnemar(b: int, c: int) -> float:
    """双侧精确 McNemar：在 H0（b、c 同分布）下 B ~ Binomial(b+c, 0.5)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def load(path: Path) -> dict[int, bool]:
    """``{idx: correct}``。idx 是样本在原 JSON 里的位置，跨文件稳定。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    eps = data.get("episodes", [])
    if not eps:
        raise ValueError(f"{path.name} 没有 episodes（是不是只存了 metrics？）")
    return {e["idx"]: bool(e.get("correct")) for e in eps}


def compare(a_path: Path, b_path: Path) -> dict:
    a, b = load(a_path), load(b_path)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("两个结果没有共同样本（idx 不重叠），不能做配对比较")
    if len(keys) != len(a) or len(keys) != len(b):
        print(f"[警告] 样本不齐：A={len(a)} B={len(b)} 共同={len(keys)}，按共同样本算")

    n = len(keys)
    ca = sum(a[k] for k in keys)
    cb = sum(b[k] for k in keys)
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    pa, pb = ca / n, cb / n
    diff = pa - pb
    se = math.sqrt(only_a + only_b) / n if (only_a + only_b) else 0.0
    z = diff / se if se else 0.0
    p = exact_mcnemar(only_a, only_b)

    # 同样效应量下，独立（未配对）比较的 SE，用于说明"配对省了多少样本"
    se_indep = math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)

    def n_for(z_target: float) -> int | None:
        """保持当前效应量，z 达到 z_target 所需样本量（配对 SE ∝ 1/sqrt(n)）。"""
        if z <= 0:
            return None
        return math.ceil(n * (z_target / z) ** 2)

    return {
        "a": a_path.name, "b": b_path.name, "n": n,
        "ea_a": pa, "ea_b": pb, "diff": diff,
        "only_a": only_a, "only_b": only_b,
        "se": se, "z": z, "p": p,
        "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se,
        "se_indep": se_indep,
        "n_sig": n_for(1.96), "n_power80": n_for(1.96 + 0.8416),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path, help="实验组（通常是更好/更靠后的配置）")
    ap.add_argument("b", type=Path, help="对照组")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not p.is_file():
            print(f"[错误] 找不到 {p}", file=sys.stderr)
            return 1

    r = compare(args.a, args.b)
    print(f"A = {r['a']}")
    print(f"B = {r['b']}")
    print(f"共同样本 n = {r['n']}")
    print(f"EA(A) = {r['ea_a']*100:.2f}%   EA(B) = {r['ea_b']*100:.2f}%   "
          f"差值 = {r['diff']*100:+.2f}pp")
    print(f"不一致对：只有 A 对 {r['only_a']}，只有 B 对 {r['only_b']}")
    print(f"配对 SE = {r['se']*100:.2f}pp（独立样本 SE = {r['se_indep']*100:.2f}pp）")
    p_txt = "p<0.001" if r["p"] < 0.001 else f"p={r['p']:.3f}"
    print(f"z = {r['z']:.2f}   精确 McNemar {p_txt}   "
          f"95% CI = [{r['ci_lo']*100:+.2f}, {r['ci_hi']*100:+.2f}]pp")
    if r["p"] < 0.05:
        print("结论：配对检验显著（p<0.05）。")
    elif r["n_sig"]:
        print(f"不显著。同一效应量下：p<0.05 约需 {r['n_sig']} 条，"
              f"80% 功效约需 {r['n_power80']} 条")
    print()
    print("判读：p>=0.05 → 只能说「方向一致的正向信号」，不能写「显著提升」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
