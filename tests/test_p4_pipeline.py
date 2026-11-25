"""P4 流水线的测试：数据格式、mask 边界、GRPO 数学、配置可解析。

这些检查**不需要 GPU、不需要 torch**，所以能放进日常 `pytest tests/`。
GPU 上的东西（显存、生成速度、rollout）由 `scripts/check_p4_env.py` 负责。

数据文件（`data/train/*.jsonl`）不进版本库，缺失时只跳过数据那一组，
其余检查照跑——保证别人 clone 下来也能验证算法部分。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.selfcheck_p4 import ROOT, run_all

DATA_DIR = ROOT / "data" / "train"


def test_mask_math_config() -> None:
    results = run_all(ROOT, groups=("mask", "grpo", "config"))
    bad = [r for r in results if not r["ok"]]
    assert not bad, f"离线检查未通过: {bad}"
    assert len(results) >= 14


@pytest.mark.skipif(not (DATA_DIR / "sft_train.jsonl").is_file(),
                    reason="data/train 未生成（跑 python scripts/build_p4_data.py）")
def test_training_data() -> None:
    results = run_all(ROOT, groups=("data",))
    bad = [r for r in results if not r["ok"]]
    assert not bad, f"数据检查未通过: {bad}"


def test_encode_truncation_never_exceeds_max_len() -> None:
    from scripts.selfcheck_p4 import FakeTokenizer
    from training.chatfmt import encode_with_spans

    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u" * 100},
        {"role": "assistant", "content": "a" * 100},
    ]
    full = encode_with_spans(FakeTokenizer(), conv)
    enc = encode_with_spans(FakeTokenizer(), conv, max_len=50)
    assert enc.length == 50
    assert len(enc.loss_mask) == 50
    assert enc.n_truncated > 0
    # 截断前后的账必须对得上（防止"截了但没记"）
    assert len(full.input_ids) == enc.length + enc.n_truncated


def test_group_advantages_shapes() -> None:
    import numpy as np

    from training.grpo import group_advantages

    adv = group_advantages(np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 3)
    assert adv.shape == (6,)
    assert np.allclose(adv.reshape(2, 3).sum(axis=1), 0, atol=1e-6)
    # 第二组全对 -> 优势全 0（不贡献梯度）
    assert np.allclose(adv[3:], 0)
