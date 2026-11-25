"""轮级奖励塑形（turn penalty / first-turn bonus）的纯逻辑测试。

背景：P5 校准发现模型准确率已达 ~86%，G=4 的组里绝大多数"全对"，
组内优势恒为 0。塑形项的作用就是让**全对组也有方差**——这个测试把这个性质钉死。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rollout.agent import Episode, Turn
from training.grpo import group_advantages, group_stats
from training.rollout import episode_reward


class _FakeEvaluator:
    def __init__(self, ok: bool):
        self.ok = ok

    def evaluate(self, db_id, gold, pred):  # noqa: ANN001
        return SimpleNamespace(correct=self.ok,
                               reason="match" if self.ok else "value_mismatch",
                               pred_error_class=None)


def _episode(n_turns: int, sql: str = "SELECT 1", stopped: str = "") -> Episode:
    ep = Episode(db_id="d", question="q", gold_sql="SELECT 1",
                 final_sql=sql, stopped_reason=stopped)
    for i in range(n_turns):
        ep.turns.append(Turn(turn=i + 1, sql=sql, success=True,
                             raw_output="```sql\nSELECT 1\n```"))
    return ep


SAMPLE = {"db_id": "d", "query": "SELECT 1"}


def test_outcome_only_is_default() -> None:
    r, info = episode_reward(_episode(3), SAMPLE, _FakeEvaluator(True))
    assert r == 1.0 and info["correct"] is True


def test_turn_penalty_and_first_turn_bonus() -> None:
    kw = dict(turn_penalty=0.1, first_turn_bonus=0.1)
    r1, _ = episode_reward(_episode(1), SAMPLE, _FakeEvaluator(True), **kw)
    r3, _ = episode_reward(_episode(3), SAMPLE, _FakeEvaluator(True), **kw)
    assert r1 == pytest.approx(1.1)
    assert r3 == pytest.approx(0.8)


def test_shaping_never_rewards_wrong_answers() -> None:
    kw = dict(turn_penalty=0.1, first_turn_bonus=0.1)
    r, _ = episode_reward(_episode(1), SAMPLE, _FakeEvaluator(False), **kw)
    assert r == 0.0
    rp, _ = episode_reward(_episode(1, sql="", stopped="parse_error"),
                           SAMPLE, _FakeEvaluator(True), turn_penalty=0.1)
    assert rp == pytest.approx(-0.2)


def test_all_correct_group_still_has_gradient() -> None:
    """同一个组全部答对，但轮数不同 -> 优势必须非 0（这是 P5 的关键前提）。"""
    rewards = np.array([1.1, 1.0, 0.8, 0.9])
    adv = group_advantages(rewards, 4)
    stats = group_stats(rewards, 4)
    assert stats["zero_var_ratio"] == 0.0
    assert np.abs(adv).max() > 0.5
    assert np.isclose(adv.sum(), 0.0, atol=1e-9)
