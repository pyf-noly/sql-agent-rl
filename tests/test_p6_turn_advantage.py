"""P6 轮级信用分配的数学与形状测试（纯 numpy，不需要 GPU/torch）。

要验的三件事：
1. **轮级优势在"全对组"里能造出信号**——这正是 outcome-only 浪费掉的那部分；
2. **全错组仍然没有信号**（轮级也救不了"4 条都错"的题，别指望它）；
3. **beta=0 时严格退化**成 P5 的 outcome-only（消融对照必须真的等价）。
"""

from __future__ import annotations

import numpy as np

from training.grpo import combine_advantages, fill_turn_advantages, turn_advantages


class TestTurnAdvantages:
    def test_all_correct_but_different_turns_gets_signal(self) -> None:
        # 4 条轨迹都做对了（outcome 全 1 → episode 优势全 0），
        # 但第 1/2/1/3 轮才做对 → 轮级优势必须非零
        first = [1, 2, 1, 3]
        n = [1, 2, 1, 3]
        adv = turn_advantages(first, n, 4)
        assert adv.shape == (4, 3)
        assert np.abs(adv).sum() > 0
        # 第 1 轮做对的两条，其第 1 轮优势应高于第 3 轮才做对的那条
        assert adv[0, 0] > adv[3, 0]

    def test_all_wrong_group_has_zero_signal(self) -> None:
        adv = turn_advantages([0, 0, 0, 0], [3, 3, 3, 3], 4)
        assert np.allclose(adv, 0.0)

    def test_beta_zero_degrades_to_outcome_only(self) -> None:
        first = [1, 2, 1, 3]
        n = [1, 2, 1, 3]
        adv = turn_advantages(first, n, 4, beta=0.0)
        assert np.allclose(adv, 0.0)
        ep = np.array([0.5, -0.5, 0.5, -0.5])
        combined = combine_advantages(ep, adv, beta=0.0)
        # 每一列都等于 episode 优势 → 与 P5 的 outcome-only 完全一致
        for t in range(combined.shape[1]):
            assert np.allclose(combined[:, t], ep)

    def test_gamma_discounts_later_turns(self) -> None:
        first = [1, 2, 2, 3]
        n = [1, 2, 2, 3]
        a1 = turn_advantages(first, n, 4, gamma=1.0)
        a2 = turn_advantages(first, n, 4, gamma=0.5)
        # 第 3 轮的优势应该被折扣得更狠
        assert abs(a2[3, 2]) < abs(a1[3, 2]) + 1e-9

    def test_group_size_mismatch_raises(self) -> None:
        try:
            turn_advantages([1, 2], [1, 2], 4)
        except ValueError:
            return
        raise AssertionError("应该抛 ValueError")

    def test_combine_shape(self) -> None:
        ep = np.array([1.0, -1.0])
        t = np.zeros((2, 3))
        t[:, 0] = [1.0, -1.0]
        out = combine_advantages(ep, t, beta=0.5)
        assert out.shape == (2, 3)
        assert np.allclose(out[:, 0], np.array([1.5, -1.5]))
        assert np.allclose(out[:, 1], np.array([1.0, -1.0]))


class TestFillTurnAdvantages:
    """把轮级优势铺到 token 上：轮内一致、轮间不同、轮外为 0。"""

    def test_fill_pattern(self) -> None:
        spans = [(2, 5), (8, 11)]
        vec = fill_turn_advantages(0.5, [1.0, -1.0], spans, 12)
        assert len(vec) == 12
        assert vec[2] == vec[4] == 1.5        # 第 1 轮：0.5 + 1.0
        assert vec[8] == vec[10] == -0.5      # 第 2 轮：0.5 - 1.0
        assert vec[0] == vec[6] == vec[11] == 0.0   # prompt/反馈/尾部

    def test_beta_zero_makes_turns_identical(self) -> None:
        spans = [(2, 5), (8, 11)]
        vec = fill_turn_advantages(0.7, [3.0, -2.0], spans, 12, beta=0.0)
        assert vec[2] == vec[8] == 0.7

    def test_spans_clipped_to_seq_len(self) -> None:
        # 截断后 span 可能超出序列长度，不能崩、不能越界写
        vec = fill_turn_advantages(0.0, [1.0], [(10, 20)], 12)
        assert len(vec) == 12
        assert vec[10] == vec[11] == 1.0

    def test_missing_turn_row_defaults_to_episode_adv(self) -> None:
        vec = fill_turn_advantages(0.3, [], [(0, 2)], 4)
        assert vec[0] == vec[1] == 0.3
