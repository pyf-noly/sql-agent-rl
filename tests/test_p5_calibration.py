"""P5 校准脚本的纯逻辑测试（不需要 GPU / torch）。

只测 ``_finalize``：筛选边界、统计口径、meta 字段。
采样部分要在卡上跑，这里不碰。
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.calibrate_rl_prompts import _finalize


def _args(tmp: Path, **kw) -> Namespace:
    base = dict(p_low=0.2, p_high=0.8, group_size=4,
                train_out=str(tmp / "kept.jsonl"))
    base.update(kw)
    return Namespace(**base)


def test_finalize_keeps_only_signal_range(tmp_path: Path) -> None:
    rows = [{"id": f"rl-{i:03d}"} for i in range(5)]
    # p: 0（全错）、0.25、0.5、0.75、1（全对）
    ps = [0.0, 0.25, 0.5, 0.75, 1.0]
    done = {
        r["id"]: {
            "id": r["id"], "db_id": "db", "question": "q", "query": "SELECT 1",
            "difficulty": "hard", "p": p, "n": 4, "n_correct": int(round(p * 4)),
            "turns_mean": 1.5,
        }
        for r, p in zip(rows, ps)
    }

    _finalize(done, rows, _args(tmp_path))

    kept = [json.loads(line) for line in
            (tmp_path / "kept.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["p"] for r in kept] == [0.25, 0.5, 0.75]

    meta = json.loads((tmp_path / "kept.meta.json").read_text(encoding="utf-8"))
    assert meta["n_calibrated"] == 5
    assert meta["n_kept"] == 3
    assert meta["all_correct"] == 1
    assert meta["all_wrong"] == 1
    assert meta["difficulty"] == {"hard": 3}

    # 理论零方差：p^G + (1-p)^G 的均值
    expect = sum(p ** 4 + (1 - p) ** 4 for p in ps) / 5
    assert abs(meta["theoretical_zero_var_ratio_at_G"] - round(expect, 3)) < 1e-9


def test_finalize_boundary_is_inclusive(tmp_path: Path) -> None:
    rows = [{"id": f"rl-{i}"} for i in range(2)]
    done = {
        r["id"]: {"id": r["id"], "db_id": "db", "question": "q", "query": "SELECT 1",
                  "difficulty": "extra", "p": p, "n": 4, "n_correct": 0, "turns_mean": 2.0}
        for r, p in zip(rows, [0.2, 0.8])
    }
    _finalize(done, rows, _args(tmp_path))
    kept = (tmp_path / "kept.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(kept) == 2, "边界 0.2 / 0.8 应当保留"
