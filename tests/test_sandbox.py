"""沙箱测试：安全、单语句、超时、结果完整性。

这一组测试是 P1 的红线。任何一条挂掉，都说明"环境不安全/不可信"，
后面训练出来的东西一律不可信。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from envs.sql_sandbox import (
    SqlSandbox,
    SqlValidationError,
    open_readonly,
    strip_trailing_semicolon,
    validate_sql,
)


class TestValidation:
    """纯校验逻辑，不碰数据库。"""

    @pytest.mark.parametrize("sql", [
        "SELECT 1",
        "SELECT 1;",
        "select 1",
        "  SELECT emp_name FROM employee WHERE salary > 100  ",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "SELECT * FROM employee e JOIN department d ON e.dept_id = d.dept_id",
        "SELECT ';' AS semi",                     # 字符串里的分号不能误判
        "SELECT 1;;",                             # 多余的空语句应被容忍
        "SELECT COUNT(*) FROM employee -- comment",
        "/* leading */ SELECT 1",
    ])
    def test_accepts_valid_select(self, sql: str) -> None:
        assert validate_sql(sql)

    @pytest.mark.parametrize("sql,reason", [
        ("", "empty_sql"),
        ("   ", "empty_sql"),
        ("DELETE FROM employee", "not_select"),
        ("UPDATE employee SET salary = 0", "not_select"),
        ("INSERT INTO employee VALUES (99, 'X', 1, 1)", "not_select"),
        ("DROP TABLE employee", "not_select"),
        ("CREATE TABLE t (a INT)", "not_select"),
        ("ALTER TABLE employee ADD COLUMN x INT", "not_select"),
        ("ATTACH DATABASE '/tmp/x.db' AS x", "not_select"),
        ("PRAGMA table_info(employee)", "not_select"),
        ("VACUUM", "not_select"),
        ("BEGIN", "not_select"),
        ("SELECT 1; DROP TABLE employee", "multi_statement"),
        ("SELECT 1; SELECT 2", "multi_statement"),
        ("SELECT * FROM employee WHERE emp_name = 'unclosed", "incomplete"),
    ])
    def test_rejects_dangerous(self, sql: str, reason: str) -> None:
        with pytest.raises(SqlValidationError) as ei:
            validate_sql(sql)
        assert reason in str(ei.value).lower()

    def test_strip_trailing_semicolon_keeps_inner(self) -> None:
        assert strip_trailing_semicolon("SELECT 1;") == "SELECT 1"
        assert strip_trailing_semicolon("SELECT 1;;; ") == "SELECT 1"
        assert strip_trailing_semicolon("SELECT ';' AS x") == "SELECT ';' AS x"


class TestReadonly:
    """物理层只读：即使绕过校验，也不能写。"""

    def test_open_readonly_rejects_write(self, tiny_db: Path) -> None:
        con = open_readonly(tiny_db)
        try:
            with pytest.raises(sqlite3.OperationalError) as ei:
                con.execute("DELETE FROM employee")
            assert "readonly" in str(ei.value).lower()
        finally:
            con.close()

    def test_sandbox_blocks_write_before_db(self, tiny_db: Path) -> None:
        sb = SqlSandbox(tiny_db)
        r = sb.run("DELETE FROM employee")
        assert not r.success
        assert r.error_class == "invalid_sql"
        # 数据必须还在
        con = open_readonly(tiny_db)
        try:
            assert con.execute("SELECT COUNT(*) FROM employee").fetchone()[0] == 4
        finally:
            con.close()


class TestExecution:
    def test_simple_select(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db).run("SELECT emp_name FROM employee ORDER BY emp_id")
        assert r.success
        assert r.columns == ["emp_name"]
        assert r.rows == [("Alice",), ("Bob",), ("Carol",), ("Dave",)]
        assert r.execution_time >= 0

    def test_join(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db).run(
            "SELECT e.emp_name, d.dept_name FROM employee e "
            "JOIN department d ON e.dept_id = d.dept_id ORDER BY e.emp_id"
        )
        assert r.success
        assert r.rows == [("Alice", "Engineering"), ("Bob", "Engineering"), ("Carol", "Sales")]

    def test_empty_result_is_success_but_empty(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db).run("SELECT * FROM employee WHERE salary > 999")
        assert r.success
        assert r.is_empty
        assert r.row_count == 0

    def test_truncation(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db, max_rows=2).run("SELECT emp_name FROM employee ORDER BY emp_id")
        assert r.success
        assert r.row_count == 2
        assert r.truncated is True

    def test_no_truncation_when_exact(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db, max_rows=4).run("SELECT emp_name FROM employee")
        assert r.truncated is False

    def test_missing_table(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db).run("SELECT * FROM employees")
        assert not r.success
        assert "no such table" in r.error_message.lower()

    def test_missing_column(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db).run("SELECT emp_nme FROM employee")
        assert not r.success
        assert "no such column" in r.error_message.lower()

    def test_syntax_error(self, tiny_db: Path) -> None:
        r = SqlSandbox(tiny_db).run("SELECT FROM WHERE")
        assert not r.success

    def test_unknown_db_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SqlSandbox(tmp_path / "nope.sqlite").run("SELECT 1")


class TestTimeout:
    """超时机制分两段验证：guard 是否真的会返回 1，以及 sandbox 是否把它翻译成 timeout。

    注意：不去 monkeypatch ``sqlite3.Connection``——在 3.15 上它是不可变类型，
    而且"注入依赖"比"打补丁改标准库"更干净，所以给 SqlSandbox 加了 connector 参数。
    """

    def test_guard_returns_nonzero_after_deadline(self) -> None:
        from envs.sql_sandbox import make_guard

        clock = {"t": 100.0}
        guard = make_guard(deadline=102.0, clock=lambda: clock["t"])

        clock["t"] = 101.0
        assert guard() == 0  # 未到截止时间，不中断
        clock["t"] = 102.5
        assert guard() == 1  # 超过 deadline，返回非 0 让 SQLite 中断
        clock["t"] = 1_000.0
        assert guard() == 1  # 幂等：之后继续返回 1

    def test_sandbox_translates_interrupt_to_timeout(self, tiny_db: Path) -> None:
        """真跑到 2s 的测试又慢又脆，直接注入一个会抛 'interrupted' 的假连接。"""
        import sqlite3

        class _InterruptingCon:
            def execute(self, sql):
                raise sqlite3.OperationalError("interrupted")

            def set_progress_handler(self, fn, n):
                return None

            def close(self):
                return None

        from envs.sql_sandbox import SqlSandbox

        sb = SqlSandbox(tiny_db, timeout=2.0, connector=lambda _p: _InterruptingCon())
        r = sb.run("SELECT 1")

        assert not r.success
        assert r.error_class == "timeout"
        assert "exceeded" in r.error_message

    def test_other_operational_error_is_db_error(self, tiny_db: Path) -> None:
        """不是 interrupted 的 OperationalError 不能被误判成超时。"""
        import sqlite3

        class _BoomCon:
            def execute(self, sql):
                raise sqlite3.OperationalError("no such table: t")

            def set_progress_handler(self, fn, n):
                return None

            def close(self):
                return None

        from envs.sql_sandbox import SqlSandbox

        sb = SqlSandbox(tiny_db, connector=lambda _p: _BoomCon())
        r = sb.run("SELECT 1")
        assert r.error_class == "db_error"
        assert "no such table" in r.error_message
