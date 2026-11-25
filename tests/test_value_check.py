"""值检查的测试：字面量抽取、存在性判断、最近邻建议、提示文本。

这个模块的价值全在"不泄漏答案"上，所以测试重点不是"能不能查到"，
而是"该报才报、不该报别乱报"——误报会把模型带偏，比没有反馈更糟。
"""

from __future__ import annotations

from envs.db_registry import DatabaseRegistry
from envs.value_check import check_literals, extract_literals, format_hints


class TestExtract:
    def test_equality_and_in(self) -> None:
        got = extract_literals(
            "SELECT * FROM employee WHERE dept_id = '2' AND emp_name IN ('Alice', 'Bob')")
        assert ("dept_id", "=", "2") in got
        assert ("emp_name", "IN", "Alice") in got
        assert ("emp_name", "IN", "Bob") in got

    def test_like(self) -> None:
        got = extract_literals("SELECT * FROM employee WHERE emp_name LIKE '%Ali%'")
        assert ("emp_name", "LIKE", "%Ali%") in got

    def test_quoted_identifier_and_alias(self) -> None:
        got = extract_literals('SELECT * FROM t WHERE t."dept_name" = \'Sales\'')
        assert ("dept_name", "=", "Sales") in got

    def test_escaped_quote(self) -> None:
        got = extract_literals("SELECT * FROM t WHERE name = 'O''Brien'")
        assert ("name", "=", "O'Brien") in got

    def test_dedup(self) -> None:
        got = extract_literals("SELECT * FROM t WHERE a = 'x' OR a = 'x'")
        assert got.count(("a", "=", "x")) == 1

    def test_no_literals(self) -> None:
        assert extract_literals("SELECT COUNT(*) FROM employee") == []


class TestCheck:
    def test_existing_value_passes(self, db_root) -> None:
        schema = DatabaseRegistry(db_root).schema("company_1")
        checks = check_literals(
            "SELECT emp_name FROM employee WHERE dept_id = '1'", schema)
        assert checks and checks[0].exists

    def test_typo_value_fails_with_suggestion(self, db_root) -> None:
        schema = DatabaseRegistry(db_root).schema("company_1")
        checks = check_literals(
            "SELECT emp_name FROM employee WHERE dept_name = 'Saless'", schema)
        # dept_name 在 department 表里，值 Sales 存在
        assert checks
        c = checks[0]
        assert c.exists is False
        assert any("Sale" in s for s in c.suggestions)

    def test_column_not_in_schema_is_skipped(self, db_root) -> None:
        schema = DatabaseRegistry(db_root).schema("company_1")
        checks = check_literals("SELECT 1 FROM employee WHERE nope = 'x'", schema)
        assert checks == []

    def test_like_partial_match_passes(self, db_root) -> None:
        schema = DatabaseRegistry(db_root).schema("company_1")
        checks = check_literals(
            "SELECT emp_name FROM employee WHERE emp_name LIKE '%lic%'", schema)
        assert checks and checks[0].exists

    def test_hint_mentions_column_and_suggestion(self, db_root) -> None:
        schema = DatabaseRegistry(db_root).schema("company_1")
        checks = check_literals(
            "SELECT emp_name FROM employee WHERE dept_name = 'Saless'", schema)
        text = format_hints(checks)
        assert "dept_name" in text
        assert "Saless" in text
        assert "Sales" in text

    def test_no_hint_when_all_exist(self, db_root) -> None:
        schema = DatabaseRegistry(db_root).schema("company_1")
        checks = check_literals(
            "SELECT emp_name FROM employee WHERE dept_name = 'Sales'", schema)
        assert format_hints(checks) == ""
