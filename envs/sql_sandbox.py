"""SQL 执行沙箱：只读、单语句、可超时。

设计目标（P1 验收点）
---------------------
1. 只读      —— 用 sqlite 的 ``mode=ro`` URI 打开，物理上不可能写库；
2. 只允许查询 —— 词法白名单只放行 SELECT / WITH，其余一律拒绝；
3. 单语句    —— 多语句（``;`` 拼接）直接拒绝，防止 ``SELECT 1; DROP TABLE x``；
4. 可超时    —— 用 ``set_progress_handler`` 做软超时；
5. 结果可复现 —— 返回列名 + 行 + 是否被截断，供后续判定 SQL 等价。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Statement

#: 允许出现的语句类型（sqlparse 的 get_type() 返回值）
ALLOWED_STMT_TYPES = {"SELECT", "UNKNOWN"}  # WITH ... SELECT 在 sqlparse 里常被识别为 UNKNOWN

#: 允许的首关键字（兜底判断，防止 UNKNOWN 里混入写操作）
ALLOWED_LEADING_KEYWORDS = {"SELECT", "WITH"}

#: 明确拒绝的关键字（即使出现在注释外也拦掉）
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "REPLACE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "TRUNCATE", "MERGE",
    "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
}


class SqlValidationError(ValueError):
    """SQL 在进入执行前就被沙箱拒绝（不是数据库报错）。"""


@dataclass(slots=True)
class QueryResult:
    """一次 SQL 执行的完整结果。"""

    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error_class: str | None = None
    error_message: str = ""
    execution_time: float = 0.0
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        """查询成功但零行。注意：这不是"错误"，但通常是错题的强信号。"""
        return self.success and len(self.rows) == 0


def strip_trailing_semicolon(sql: str) -> str:
    """去掉末尾分号（允许一个或多个），保留语句内部的分号不动。

    ``SELECT ';'`` 这种字符串里的分号不能被误删，所以只处理结尾。
    """
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _is_comment_token(tok) -> bool:
    """判断一个 token 是不是注释。

    不能只用 ``tok.ttype in T.Comment``：sqlparse 对**前导块注释**的处理不一致，
    ``/* c */ SELECT 1`` 的第一个 token 是 ttype=None 的裸 token，值是 ``/* c */``。
    所以同时按 ttype 和字符串前缀判断。
    """
    v = tok.value
    return tok.ttype in T.Comment or v.startswith("/*") or v.startswith("--")


def _leading_keyword(stmt: Statement) -> str:
    """取第一条真正的关键字，跳过注释和空白。

    sqlparse 会把前导注释当成第一个 token（``/* c */ SELECT`` 的 lead 会是
    ``/* c */``），不跳过就会把合法查询误判成非法。
    """
    for tok in stmt.tokens:
        if tok.is_whitespace or _is_comment_token(tok):
            continue
        return tok.value.upper()
    return ""


def _first_keyword_anywhere(stmt: Statement) -> str:
    """兜底：在整棵 token 树里找第一个关键字。

    用于 ``WITH ... SELECT`` 这种 ``get_type()`` 返回 UNKNOWN 的语句。
    """
    for tok in stmt.flatten():
        if tok.is_whitespace or _is_comment_token(tok):
            continue
        if tok.ttype in T.Keyword:
            return tok.value.upper()
    return ""


def _ends_with_line_comment(sql: str) -> bool:
    """判断 SQL 最后一行（引号外）是否以 ``--`` 开头，即结尾是行注释。

    为什么要单独判断：``sqlite3.complete_statement("SELECT 1 -- note")``
    返回 **False**——它要求行注释后面必须有换行。这是本模块唯一需要"修整"的
    书写习惯，不修整就会把合法查询误判成"语句不完整"。
    """
    last = sql.rstrip().splitlines()[-1] if sql.strip() else ""
    in_quote = False
    for i in range(len(last) - 1):
        ch = last[i]
        if ch == "'":
            in_quote = not in_quote
        elif ch == "-" and last[i + 1] == "-" and not in_quote:
            return True
    return False


def _normalize_for_validation(sql: str) -> str:
    """把"只是书写不规范"的语句整理成可安全校验的形式。

    只处理一件事：**未闭合的块注释**（``SELECT 1 /* note``）——这是真的不完整，
    把残缺注释砍掉再校验。

    行注释结尾（``SELECT 1 -- note``）不在这里处理，因为要交给
    :func:`_terminate_for_validation`：``complete_statement`` 要求行注释后有换行，
    直接补分号会被注释吞掉。
    """
    s = sql.strip()
    last_open = s.rfind("/*")
    if last_open != -1 and s.find("*/", last_open) == -1:
        return s[:last_open].rstrip()
    return s


def _terminate_for_validation(sql: str) -> str:
    """给语句补一个"能正确结束"的终止符。

    - 以行注释结尾时，补 ``\\n;``（换成行，分号才不在注释里）；
    - 否则补 ``;``。

    只用于 ``complete_statement`` 判断，不改变要执行的 SQL 主体。
    """
    if _ends_with_line_comment(sql):
        return sql.rstrip() + "\n;"
    return sql.rstrip().rstrip(";").rstrip() + ";"


def validate_sql(sql: str) -> str:
    """校验并返回去掉尾分号的 SQL；不合法抛 ``SqlValidationError``。"""
    if not sql or not sql.strip():
        raise SqlValidationError("empty_sql: 输入为空")

    sql = _normalize_for_validation(sql)

    if not sqlite3.complete_statement(_terminate_for_validation(sql)):
        raise SqlValidationError("incomplete_sql: 语句不完整（引号或括号未闭合）")

    body = strip_trailing_semicolon(sql)

    # 关键：去掉尾分号后，若仍能被 complete_statement 认可，说明后面还有第二条语句
    if not sqlite3.complete_statement(_terminate_for_validation(body)):
        raise SqlValidationError("incomplete_sql: 语句不完整（引号或括号未闭合）")

    parts = [p for p in sqlparse.split(body) if p.strip()]
    if len(parts) != 1:
        raise SqlValidationError(
            f"multi_statement: 只允许一条语句，检测到 {len(parts)} 条"
        )
    body = parts[0]

    try:
        parsed = sqlparse.parse(body)
    except RecursionError as e:  # 畸形 SQL 可能把解析器打爆
        raise SqlValidationError("unparseable_sql: 无法解析") from e

    if not parsed:
        raise SqlValidationError("empty_sql: 解析结果为空")

    stmt = parsed[0]
    lead = _leading_keyword(stmt)

    # 白名单只看首关键字。不用 get_type() 做主判据，因为它对 WITH ... SELECT
    # 返回 UNKNOWN，会把合法 CTE 误杀；get_type() 只在下面做兜底一致性检查。
    if lead not in ALLOWED_LEADING_KEYWORDS:
        raise SqlValidationError(
            f"not_select: 只允许 SELECT/WITH 查询，当前为 {lead or stmt.get_type()}"
        )

    # 词法级别的兜底：即使首关键字是 SELECT，也不许出现写操作关键字
    for tok in stmt.flatten():
        if tok.ttype in T.Keyword and tok.value.upper() in FORBIDDEN_KEYWORDS:
            raise SqlValidationError(
                f"forbidden_keyword: 不允许出现关键字 {tok.value.upper()}"
            )

    return body


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """以只读模式打开 sqlite 文件。

    ``mode=ro`` 是 SQLite 的 URI 参数，打开后任何写操作都会报
    ``attempt to write a readonly database``——这是第一道物理防线。
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"数据库不存在: {path}")
    # Windows 盘符路径（E:/x/y.db）与 POSIX 路径（/x/y.db）在 URI 里都能直接用
    uri = f"file:{path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=5.0)
    return con


def make_guard(deadline: float, clock=time.perf_counter) -> Callable[[], int]:
    """构造 progress handler：到点返回 1 让 SQLite 中断当前语句。

    单独拎出来是为了可单测——否则只能靠"真的跑满 2 秒"来验证，又慢又脆。
    ``clock`` 可注入，测试里换成假时钟就能精确控制。
    """

    def guard() -> int:
        return 1 if clock() > deadline else 0

    return guard


class SqlSandbox:
    """执行器。每次执行都新开连接，避免状态污染。

    Parameters
    ----------
    db_path:
        sqlite 文件路径。
    timeout:
        软超时秒数。通过 progress handler 在 VM 指令级检查，
        对 ``WITH RECURSIVE`` 死循环非常有效。
    max_rows:
        结果行数上限，超过即截断并置 ``truncated=True``。
        防的是「查出 100 万行把内存打爆」。
    connector:
        连接工厂，默认 :func:`open_readonly`。允许注入是为了测试
        （测试里换成一个会抛 ``interrupted`` 的假连接，不用真跑 2 秒）。
    """

    def __init__(
        self,
        db_path: str | Path,
        timeout: float = 2.0,
        max_rows: int = 1000,
        connector: Callable[[str | Path], sqlite3.Connection] = open_readonly,
    ):
        self.db_path = Path(db_path)
        self.timeout = float(timeout)
        self.max_rows = int(max_rows)
        self._connector = connector

    def run(self, sql: str) -> QueryResult:
        """执行一条 SQL，永不抛异常，所有失败都体现在返回值里。"""
        t0 = time.perf_counter()

        try:
            body = validate_sql(sql)
        except SqlValidationError as e:
            return QueryResult(
                success=False,
                error_class="invalid_sql",
                error_message=str(e),
                execution_time=time.perf_counter() - t0,
            )

        con: sqlite3.Connection | None = None
        try:
            con = self._connector(self.db_path)
            deadline = time.perf_counter() + self.timeout

            con.set_progress_handler(make_guard(deadline), 2000)
            cur = con.execute(body)

            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            if truncated:
                rows = rows[: self.max_rows]

            return QueryResult(
                success=True,
                columns=cols,
                rows=[tuple(r) for r in rows],
                execution_time=time.perf_counter() - t0,
                truncated=truncated,
            )
        except sqlite3.OperationalError as e:
            msg = str(e)
            if "interrupted" in msg.lower():
                return QueryResult(
                    success=False,
                    error_class="timeout",
                    error_message=f"query exceeded {self.timeout:g}s and was interrupted",
                    execution_time=time.perf_counter() - t0,
                )
            return QueryResult(
                success=False,
                error_class="db_error",
                error_message=msg,
                execution_time=time.perf_counter() - t0,
            )
        except sqlite3.DatabaseError as e:
            return QueryResult(
                success=False,
                error_class="db_error",
                error_message=str(e),
                execution_time=time.perf_counter() - t0,
            )
        finally:
            if con is not None:
                try:
                    con.set_progress_handler(None, 0)
                except Exception:
                    pass
                con.close()
