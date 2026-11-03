"""错误分类 + 反馈模板测试。

重点验证两件事：
1. "ambiguous column name" 不能被 no_such_column 抢先匹配（顺序 bug 的高发区）；
2. 反馈文本必须**不含** gold SQL / 正确答案（防泄漏）。
"""

from __future__ import annotations

import pytest

from envs.error_taxonomy import ErrorClass, classify, classify_result, extract_offender
from envs.feedback import FeedbackConfig, FeedbackFormatter
from envs.sql_sandbox import QueryResult


class TestClassify:
    @pytest.mark.parametrize("msg,expected", [
        ("near \"FROM\": syntax error", ErrorClass.SYNTAX),
        ("incomplete input", ErrorClass.SYNTAX),
        ("no such table: employees", ErrorClass.NO_SUCH_TABLE),
        ("no such column: emp_nme", ErrorClass.NO_SUCH_COLUMN),
        ("ambiguous column name: id", ErrorClass.AMBIGUOUS_COLUMN),
        ("query exceeded 2s and was interrupted", ErrorClass.TIMEOUT),
        ("empty result", ErrorClass.EMPTY_RESULT),
        ("unable to open database file", ErrorClass.DB_ERROR),
    ])
    def test_map_messages(self, msg: str, expected: ErrorClass) -> None:
        assert classify("db_error", msg) is expected

    def test_ambiguous_wins_over_no_such_column(self) -> None:
        # "ambiguous column name: id" 里没有 "no such column"，但顺序仍要保证
        assert classify("db_error", "ambiguous column name: id") is ErrorClass.AMBIGUOUS_COLUMN

    def test_success_with_rows_is_unknown(self) -> None:
        assert classify(None, None, is_empty=False) is ErrorClass.UNKNOWN

    def test_success_empty_is_empty_result(self) -> None:
        assert classify(None, None, is_empty=True) is ErrorClass.EMPTY_RESULT

    def test_invalid_sql_passthrough(self) -> None:
        assert classify("invalid_sql", "multi_statement: x") is ErrorClass.INVALID_SQL

    @pytest.mark.parametrize("msg,offender", [
        ("no such column: t.emp_name", "t.emp_name"),
        ("no such table: employees", "employees"),
        ("ambiguous column name: id", "id"),
        ("near \"X\": syntax error", None),
    ])
    def test_extract_offender(self, msg: str, offender) -> None:
        assert extract_offender(msg) == offender

    def test_is_schema_error(self) -> None:
        c = classify_result(False, "db_error", "no such table: t", is_empty=False)
        assert c.is_schema_error
        c2 = classify_result(False, "db_error", "near \"x\": syntax error", is_empty=False)
        assert not c2.is_schema_error


def _result(**kw) -> QueryResult:
    base = dict(success=True, columns=["a"], rows=[(1,)], error_class=None, error_message="")
    base.update(kw)
    return QueryResult(**base)


class TestFeedback:
    def setup_method(self) -> None:
        self.fmt = FeedbackFormatter()
        self.tables = ["department", "employee"]
        self.columns = {
            "department": ["dept_id", "dept_name", "budget"],
            "employee": ["emp_id", "emp_name", "dept_id", "salary"],
        }

    def test_success_preview(self) -> None:
        r = _result(columns=["emp_name"], rows=[("Alice",), ("Bob",)])
        text = self.fmt.render_success(r)
        assert "succeeded" in text
        assert "Alice" in text

    def test_success_empty_warns(self) -> None:
        text = self.fmt.render_success(_result(rows=[]))
        assert "EMPTY" in text

    def test_no_such_table_lists_tables(self) -> None:
        c = classify_result(False, "db_error", "no such table: employees", is_empty=False)
        text = self.fmt.render(_result(success=False, error_message=c.raw_message),
                               c, table_names=self.tables)
        assert "employees" in text
        assert "employee" in text

    def test_no_such_column_scoped_to_table(self) -> None:
        c = classify_result(False, "db_error", "no such column: employee.emp_nme", is_empty=False)
        text = self.fmt.render(_result(success=False, error_message=c.raw_message),
                               c, columns=self.columns)
        assert "emp_nme" in text
        assert "emp_name" in text
        # 只列相关表的列，不应该把 department 的列也倒出来
        assert "budget" not in text

    def test_ambiguous_lists_owners(self) -> None:
        c = classify_result(False, "db_error", "ambiguous column name: dept_id", is_empty=False)
        text = self.fmt.render(_result(success=False, error_message=c.raw_message),
                               c, columns=self.columns)
        assert "department" in text and "employee" in text
        assert "qualify" in text.lower()

    def test_invalid_sql_message(self) -> None:
        c = classify_result(False, "invalid_sql", "not_select: 只允许 SELECT", is_empty=False)
        text = self.fmt.render(_result(success=False, error_message=c.raw_message), c)
        assert "SELECT" in text

    def test_timeout_mentions_causes(self) -> None:
        c = classify_result(False, "timeout", "exceeded 2s", is_empty=False)
        text = self.fmt.render(_result(success=False, error_message="exceeded 2s"), c)
        assert "cartesian" in text.lower() or "JOIN" in text

    def test_schema_hint_can_be_disabled(self) -> None:
        """消融开关：关掉 schema 提示后，反馈里不应该再出现表名列表。"""
        fmt = FeedbackFormatter(FeedbackConfig(include_schema_hint=False))
        c = classify_result(False, "db_error", "no such table: employees", is_empty=False)
        text = fmt.render(_result(success=False), c, table_names=self.tables)
        assert "department" not in text

    @pytest.mark.parametrize("gold", ["SELECT COUNT(*) FROM employee", "Alice", "42"])
    def test_no_gold_leakage(self, gold: str) -> None:
        """反馈文本里绝不能出现 gold SQL / 答案，否则就是数据泄漏。"""
        cases = [
            classify_result(False, "db_error", "no such table: x", is_empty=False),
            classify_result(False, "db_error", "no such column: employee.y", is_empty=False),
            classify_result(False, "db_error", "near \"Z\": syntax error", is_empty=False),
            classify_result(True, None, "", is_empty=True),
        ]
        for c in cases:
            text = self.fmt.render(_result(success=c.category is ErrorClass.EMPTY_RESULT),
                                   c, table_names=self.tables, columns=self.columns)
            assert gold not in text

    def test_deterministic(self) -> None:
        c = classify_result(False, "db_error", "no such table: x", is_empty=False)
        a = self.fmt.render(_result(success=False), c, table_names=self.tables)
        b = self.fmt.render(_result(success=False), c, table_names=self.tables)
        assert a == b
