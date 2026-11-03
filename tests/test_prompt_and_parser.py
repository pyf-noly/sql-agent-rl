"""Prompt 与 SQL 解析测试。

解析器是"模型输出 → 可执行 SQL"的唯一通道，它抽错等于模型答错，
所以边界情况（无代码块、多代码块、裸 SQL、空输出）必须全覆盖。
"""

from __future__ import annotations

import pytest

from envs.db_registry import DbSchema, TableInfo
from rollout.parse_sql import extract_sql, parse_sql
from rollout.prompt import PromptConfig, build_feedback_message, build_messages, build_user_prompt


@pytest.fixture()
def schema(tmp_path) -> DbSchema:
    return DbSchema(
        db_id="demo",
        path=tmp_path / "demo.sqlite",
        tables=(
            TableInfo("singer", ("singer_id", "name", "age")),
            TableInfo("concert", ("concert_id", "singer_id", "year")),
        ),
    )


class TestPrompt:
    def test_contains_schema_question_rules(self, schema) -> None:
        p = build_user_prompt("How many singers?", schema)
        assert "CREATE TABLE singer" in p
        assert "How many singers?" in p
        assert "```sql" in p

    def test_messages_shape(self, schema) -> None:
        msgs = build_messages("q", schema)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_schema_truncation(self, schema) -> None:
        cfg = PromptConfig(max_schema_chars=40)
        p = build_user_prompt("q", schema, cfg)
        assert "(schema truncated)" in p

    def test_ddl_can_be_disabled(self, schema) -> None:
        cfg = PromptConfig(include_ddl=False)
        p = build_user_prompt("q", schema, cfg)
        assert "CREATE TABLE" not in p

    def test_feedback_message_has_prev_sql(self) -> None:
        m = build_feedback_message("SELECT 1", "no such column: x")
        assert "SELECT 1" in m["content"]
        assert "no such column: x" in m["content"]

    def test_system_prompt_forbids_prose(self, schema) -> None:
        sys_msg = build_messages("q", schema)[0]["content"]
        assert "code block" in sys_msg
        assert "No other text" in sys_msg


class TestParseSql:
    @pytest.mark.parametrize("text,expected", [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```sqlite\nSELECT 1;\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("Here you go:\n```sql\nSELECT name FROM singer\n```\nDone.", "SELECT name FROM singer"),
        ("SELECT 1;", "SELECT 1"),
        ("  select 1  ", "select 1"),
    ])
    def test_extracts(self, text: str, expected: str) -> None:
        assert extract_sql(text) == expected

    def test_prefers_last_block_with_select(self) -> None:
        text = "First:\n```sql\nEXPLAIN\n```\nActually:\n```sql\nSELECT 1\n```"
        assert extract_sql(text) == "SELECT 1"

    def test_no_fence_but_select_in_prose(self) -> None:
        text = "The query is SELECT count(*) FROM singer because reasons."
        parsed = parse_sql(text)
        assert parsed.source == "bare"
        assert parsed.sql.startswith("SELECT count(*)")

    @pytest.mark.parametrize("text", ["", "   ", None, "Sorry, I cannot answer."])
    def test_unparseable_returns_empty(self, text) -> None:
        parsed = parse_sql(text)
        assert parsed.sql == ""
        assert parsed.source == "none"
        assert not parsed.ok

    def test_strips_multiple_semicolons(self) -> None:
        assert extract_sql("```sql\nSELECT 1;;;\n```") == "SELECT 1"

    def test_keeps_semicolon_inside_string(self) -> None:
        assert extract_sql("```sql\nSELECT ';' AS x\n```") == "SELECT ';' AS x"

    def test_raw_preserved_for_debugging(self) -> None:
        text = "garbage without sql"
        assert parse_sql(text).raw == text

    def test_multiline_sql(self) -> None:
        text = "```sql\nSELECT a,\n       b\nFROM t\nWHERE x = 1\n```"
        assert extract_sql(text).count("\n") >= 3
