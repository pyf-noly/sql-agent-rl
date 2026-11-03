"""工具反馈模板：把执行结果变成回灌给模型的一段文本。

这是"环境"和"模型"之间唯一的文字接口，改这里等于改环境的全部信息量。
所以单独成文件，且所有模板都可配置——后面做消融（只回传报错 vs 回传结构化提示）
只需要换这里，不动沙箱。

设计原则
--------
1. **短**：每条反馈控制在 ~120 token 内，否则多轮会把上下文撑爆；
2. **可执行**：告诉模型"错在哪、正确选项有哪些"，而不是只说 "error"；
3. **不泄答案**：绝不把 gold SQL 或正确结果塞进去，否则就是数据泄漏；
4. **确定性**：同样的输入必须给同样的文本（便于 RL 复现）。
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.error_taxonomy import ClassifiedError, ErrorClass
from envs.sql_sandbox import QueryResult

#: 提示里最多列多少个标识符，防止宽表把上下文撑爆
MAX_IDENTIFIERS = 20

#: 结果预览最多给几行
MAX_PREVIEW_ROWS = 3


@dataclass(slots=True)
class FeedbackConfig:
    """反馈风格开关，后面做消融时改这里。"""

    include_schema_hint: bool = True     # schema 类错误时附上可用的表/列
    include_result_preview: bool = True  # 成功时附上前几行结果
    max_identifiers: int = MAX_IDENTIFIERS
    max_preview_rows: int = MAX_PREVIEW_ROWS


def _clip(items: list[str], n: int) -> str:
    if not items:
        return "(none)"
    head = items[:n]
    s = ", ".join(head)
    if len(items) > n:
        s += f", ... (+{len(items) - n} more)"
    return s


def _fmt_table_names(tables: list[str]) -> str:
    return _clip(tables, MAX_IDENTIFIERS)


def _fmt_row(row: tuple) -> str:
    def _one(v: object) -> str:
        if v is None:
            return "NULL"
        s = str(v)
        return s if len(s) <= 30 else s[:27] + "..."

    return "(" + ", ".join(_one(v) for v in row) + ")"


class FeedbackFormatter:
    """把 :class:`QueryResult` + 分类结果渲染成反馈文本。"""

    def __init__(self, cfg: FeedbackConfig | None = None):
        self.cfg = cfg or FeedbackConfig()

    def render(
        self,
        result: QueryResult,
        classified: ClassifiedError,
        *,
        table_names: list[str] | None = None,
        columns: dict[str, list[str]] | None = None,
    ) -> str:
        """Parameters
        ----------
        table_names:
            当前库的所有表名（用于 no_such_table 提示）。
        columns:
            ``{table: [column, ...]}``（用于 no_such_column / ambiguous 提示）。
        """
        cat = classified.category

        if cat is ErrorClass.INVALID_SQL:
            return self._invalid_sql(classified)
        if cat is ErrorClass.SYNTAX:
            return self._syntax(classified)
        if cat is ErrorClass.NO_SUCH_TABLE:
            return self._no_such_table(classified, table_names)
        if cat is ErrorClass.NO_SUCH_COLUMN:
            return self._no_such_column(classified, columns)
        if cat is ErrorClass.AMBIGUOUS_COLUMN:
            return self._ambiguous(classified, columns)
        if cat is ErrorClass.TIMEOUT:
            return self._timeout(result)
        if cat is ErrorClass.EMPTY_RESULT:
            return self._empty()
        return self._generic(classified)

    # ---------------- 成功 ----------------

    def render_success(self, result: QueryResult) -> str:
        n = result.row_count
        head = f"Execution succeeded: {n} row{'s' if n != 1 else ''}"
        if result.truncated:
            head += f" (truncated at {self.cfg.max_preview_rows} for preview)"
        if n == 0:
            head += ". The result is EMPTY — double-check your WHERE / JOIN conditions."
            return head
        if not self.cfg.include_result_preview:
            return head + "."
        lines = [head + ".", f"Columns: {_clip(result.columns, MAX_IDENTIFIERS)}"]
        for row in result.rows[: self.cfg.max_preview_rows]:
            lines.append(f"  {_fmt_row(row)}")
        return "\n".join(lines)

    # ---------------- 各类失败 ----------------

    def _invalid_sql(self, c: ClassifiedError) -> str:
        reason = c.raw_message.split(":", 1)[-1].strip()
        return (
            "Your answer was rejected before execution.\n"
            f"Reason: {reason}\n"
            "You must reply with exactly one SELECT (or WITH ... SELECT) statement "
            "inside a ```sql code block. No INSERT/UPDATE/DELETE/PRAGMA, no multiple statements."
        )

    def _syntax(self, c: ClassifiedError) -> str:
        return (
            "SQL syntax error from the database.\n"
            f"Database says: {c.raw_message.strip()}\n"
            "Fix the syntax and resubmit. Common causes: missing comma between columns, "
            "unbalanced parentheses/quotes, wrong clause order (WHERE before GROUP BY, "
            "HAVING after GROUP BY, ORDER BY last), or using a reserved word as an identifier."
        )

    def _no_such_table(self, c: ClassifiedError, table_names: list[str] | None) -> str:
        msg = f"Table `{c.offender}` does not exist in this database."
        if self.cfg.include_schema_hint and table_names:
            msg += f"\nAvailable tables: {_fmt_table_names(table_names)}"
        msg += "\nUse only the tables listed in the schema above."
        return msg

    def _no_such_column(self, c: ClassifiedError, columns: dict[str, list[str]] | None) -> str:
        msg = f"Column `{c.offender}` does not exist."
        if self.cfg.include_schema_hint and columns:
            # 找出 offender 里可能的表前缀，只列相关表的列，省 token
            owner = c.offender.split(".")[0] if c.offender and "." in c.offender else None
            if owner and owner in columns:
                msg += f"\nColumns of table `{owner}`: {_clip(columns[owner], self.cfg.max_identifiers)}"
            else:
                flat = []
                for t, cols in columns.items():
                    flat.extend(f"{t}.{col}" for col in cols)
                msg += f"\nAvailable columns: {_clip(flat, self.cfg.max_identifiers)}"
        msg += "\nUse the exact column names (case-sensitive names in SQLite may need quoting)."
        return msg

    def _ambiguous(self, c: ClassifiedError, columns: dict[str, list[str]] | None) -> str:
        msg = f"Column `{c.offender}` is ambiguous — it appears in more than one joined table."
        if self.cfg.include_schema_hint and columns and c.offender:
            owners = [t for t, cols in columns.items() if c.offender in cols]
            if owners:
                msg += f"\nIt exists in: {_clip(owners, self.cfg.max_identifiers)}"
        msg += "\nQualify it with the table name or alias, e.g. `t.col`."
        return msg

    def _timeout(self, result: QueryResult) -> str:
        return (
            f"Query exceeded the {result.execution_time:.1f}s execution budget and was cancelled.\n"
            "Likely causes: a cartesian product from a missing JOIN condition, "
            "or `WITH RECURSIVE` without a termination condition. "
            "Add the JOIN condition / a LIMIT, or restructure the query."
        )

    def _empty(self) -> str:
        return (
            "Execution succeeded but returned 0 rows.\n"
            "This usually means a wrong filter value, a JOIN on non-matching keys, "
            "or comparing against a value that does not exist in the data. "
            "Re-check the literals in WHERE and the JOIN keys."
        )

    def _generic(self, c: ClassifiedError) -> str:
        return (
            "Execution failed with a database error.\n"
            f"Database says: {c.raw_message.strip()}\n"
            "Fix the query and resubmit."
        )
