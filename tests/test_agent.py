"""Agent 交互测试：用假 LLM 驱动，不花钱、不联网。

这组测试的价值在于：真正调 API 之前，先证明"多轮循环"本身是对的——
反馈有没有回灌、轮数有没有守住、执行成功会不会提前停、解析失败怎么处理。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from envs.db_registry import DatabaseRegistry
from rollout.agent import SqlAgent
from rollout.feedback_stub import StubLLM  # 见下


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


class TestSingleTurn:
    def test_correct_sql(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM(["```sql\nSELECT count(*) FROM singer\n```"])
        ep = SqlAgent(llm, reg, max_turns=1).run(
            {"db_id": "demo", "question": "how many singers?", "query": "SELECT count(*) FROM singer"})
        assert ep.n_turns == 1
        assert ep.final_sql == "SELECT count(*) FROM singer"
        assert ep.stopped_reason == "success"

    def test_wrong_column_stops_at_max_turns(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM(["```sql\nSELECT nme FROM singer\n```"] * 3)
        ep = SqlAgent(llm, reg, max_turns=3).run(
            {"db_id": "demo", "question": "q", "query": "SELECT name FROM singer"})
        assert ep.n_turns == 3
        assert ep.stopped_reason == "max_turns"
        assert all("no such column" in t.error_message.lower() for t in ep.turns)


class TestMultiTurnRepair:
    def test_second_turn_gets_feedback_and_fixes(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM([
            "```sql\nSELECT nme FROM singer\n```",   # 第 1 轮：列名错
            "```sql\nSELECT name FROM singer\n```",  # 第 2 轮：修好
        ])
        ep = SqlAgent(llm, reg, max_turns=3).run(
            {"db_id": "demo", "question": "names?", "query": "SELECT name FROM singer"})

        assert ep.n_turns == 2
        assert ep.stopped_reason == "success"
        assert ep.final_sql == "SELECT name FROM singer"

        # 第 2 轮请求里必须带上一轮的反馈
        second_call_msgs = llm.calls[1]
        joined = " ".join(m["content"] for m in second_call_msgs)
        assert "nme" in joined
        assert "no such column" in joined.lower()

    def test_feedback_does_not_leak_gold(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM(["```sql\nSELECT nope FROM singer\n```"] * 2)
        SqlAgent(llm, reg, max_turns=2).run(
            {"db_id": "demo", "question": "q", "query": "SELECT name FROM singer"})
        joined = " ".join(m["content"] for m in llm.calls[1])
        # gold SQL 的具体形态不能出现在反馈里
        assert "SELECT name FROM singer" not in joined

    def test_empty_result_is_success_but_feedback_warns(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM(["```sql\nSELECT name FROM singer WHERE age > 999\n```",
                       "```sql\nSELECT name FROM singer WHERE age > 20\n```"])
        ep = SqlAgent(llm, reg, max_turns=3, stop_on_success=False).run(
            {"db_id": "demo", "question": "q", "query": "SELECT name FROM singer"})
        assert ep.n_turns == 3  # stop_on_success=False，跑满
        assert "EMPTY" in ep.turns[0].feedback


class TestParseFailure:
    def test_no_sql_gives_feedback_and_retries(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM([
            "I cannot answer that.",                      # 无 SQL
            "```sql\nSELECT count(*) FROM singer\n```",   # 修好
        ])
        ep = SqlAgent(llm, reg, max_turns=3).run(
            {"db_id": "demo", "question": "q", "query": "SELECT count(*) FROM singer"})
        assert ep.turns[0].error_class == "parse_error"
        assert ep.n_turns == 2
        assert ep.stopped_reason == "success"


class TestSafety:
    def test_dangerous_sql_is_rejected_by_sandbox(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM(["```sql\nDROP TABLE singer\n```"])
        ep = SqlAgent(llm, reg, max_turns=1).run(
            {"db_id": "demo", "question": "q", "query": "SELECT 1"})
        assert ep.turns[0].error_class == "invalid_sql"
        # 表还在
        con = sqlite3.connect(reg.db_path("demo"))
        try:
            assert con.execute("SELECT count(*) FROM singer").fetchone()[0] == 3
        finally:
            con.close()

    def test_api_error_recorded_not_raised(self, reg: DatabaseRegistry) -> None:
        llm = StubLLM([RuntimeError("boom")])
        ep = SqlAgent(llm, reg, max_turns=2).run(
            {"db_id": "demo", "question": "q", "query": "SELECT 1"})
        assert ep.stopped_reason == "api_error"
        assert "boom" in ep.api_error
