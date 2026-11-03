"""多轮"判对才停"模式测试：验证 verifier 只决定是否继续、不会泄漏答案。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from envs.db_registry import DatabaseRegistry
from rollout.agent import SqlAgent
from rollout.evaluator import Evaluator
from rollout.feedback_stub import StubLLM


@pytest.fixture()
def reg(tmp_path: Path) -> DatabaseRegistry:
    root = tmp_path / "database"
    d = root / "demo"
    d.mkdir(parents=True)
    con = sqlite3.connect(d / "demo.sqlite")
    con.executescript("""
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INT);
        INSERT INTO singer VALUES (1,'Alice',30),(2,'Bob',25),(3,'Carol',40);
    """)
    con.commit()
    con.close()
    return DatabaseRegistry(root)


SAMPLE = {"db_id": "demo", "question": "how many?", "query": "SELECT count(*) FROM singer"}


class TestUntilCorrect:
    def test_stops_when_correct(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM(["```sql\nSELECT count(*) FROM singer\n```"])
        ep = SqlAgent(llm, reg, max_turns=3, stop_on_success=False,
                      verifier=Evaluator(reg), stop_on_correct=True).run(SAMPLE)
        assert ep.n_turns == 1
        assert ep.stopped_reason == "correct"
        assert ep.turns[0].correct is True

    def test_continues_while_wrong(self, reg: DatabaseRegistry) -> None:
        # 三次都"能执行但答错"，跑满三轮
        llm = StubLLM(["```sql\nSELECT name FROM singer\n```"] * 3)
        ep = SqlAgent(llm, reg, max_turns=3, stop_on_success=False,
                      verifier=Evaluator(reg), stop_on_correct=True).run(SAMPLE)
        assert ep.n_turns == 3
        assert ep.stopped_reason == "max_turns"
        assert all(t.correct is False for t in ep.turns)

    def test_second_turn_can_fix(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM([
            "```sql\nSELECT name FROM singer\n```",   # 能执行，但结果不对
            "```sql\nSELECT count(*) FROM singer\n```",  # 对了
        ])
        ep = SqlAgent(llm, reg, max_turns=3, stop_on_success=False,
                      verifier=Evaluator(reg), stop_on_correct=True).run(SAMPLE)
        assert ep.n_turns == 2
        assert ep.stopped_reason == "correct"
        assert ep.turns[-1].correct is True

    def test_correctness_not_leaked_in_feedback(self, reg: DatabaseRegistry) -> None:
        """最关键的一条：判对判错的信息不能出现在回灌文本里。

        注意不能简单 grep "correct"——提示语里的 "Write a corrected query"
        是正常指令。要查的是"判定性表述"和"答案内容"。
        """
        llm = StubLLM([
            "```sql\nSELECT name FROM singer\n```",
            "```sql\nSELECT count(*) FROM singer\n```",
        ])
        SqlAgent(llm, reg, max_turns=3, stop_on_success=False,
                 verifier=Evaluator(reg), stop_on_correct=True).run(SAMPLE)

        joined = " ".join(m["content"] for m in llm.calls[1]).lower()
        for forbidden in [
            "is correct", "is incorrect", "answer is right", "answer is wrong",
            "gold sql", "expected answer", "reference answer",
            "select count(*) from singer",  # gold 本体
        ]:
            assert forbidden not in joined, f"反馈里出现泄漏: {forbidden}"

    def test_requires_verifier(self, reg: DatabaseRegistry) -> None:
        with pytest.raises(ValueError, match="verifier"):
            SqlAgent(StubLLM(["x"]), reg, stop_on_correct=True)

    def test_execution_error_also_continues(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM([
            "```sql\nSELECT nope FROM singer\n```",
            "```sql\nSELECT count(*) FROM singer\n```",
        ])
        ep = SqlAgent(llm, reg, max_turns=3, stop_on_success=False,
                      verifier=Evaluator(reg), stop_on_correct=True).run(SAMPLE)
        assert ep.n_turns == 2
        assert ep.turns[0].success is False
        assert ep.turns[0].correct is None  # 没执行成功就不判对错
