"""错误分类：把 sqlite 的报错文本映射成稳定的类别。

为什么需要分类？
---------------
1. 反馈模板要按类别给不同提示（"列名错了"和"结果为空"该给的提示完全不同）；
2. 后面分析失败样本时，要按"错误类型 × 是否被 RL 修复"做交叉表；
3. 类别是有限的，可以做成 reward 的输入（例如「从 schema 类错误中恢复」加分）。

约定：只做**文本映射**，不执行任何 SQL，纯函数、可单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorClass(str, Enum):
    INVALID_SQL = "invalid_sql"        # 沙箱在进数据库前就拒绝（非法语句/多语句）
    SYNTAX = "syntax"                  # 语法错误
    NO_SUCH_COLUMN = "no_such_column"  # 列不存在
    NO_SUCH_TABLE = "no_such_table"    # 表不存在
    AMBIGUOUS_COLUMN = "ambiguous_column"  # 列名歧义
    TIMEOUT = "timeout"                # 超时
    EMPTY_RESULT = "empty_result"      # 执行成功但零行（不是报错，但常是错题）
    DB_ERROR = "db_error"              # 其他数据库错误
    UNKNOWN = "unknown"


#: 顺序很重要：更具体的模式必须排在前面。
#: 例如 "ambiguous column name" 同时包含 "column"，必须比 no_such_column 先匹配。
_PATTERNS: list[tuple[ErrorClass, re.Pattern[str]]] = [
    (ErrorClass.INVALID_SQL, re.compile(r"^(empty_sql|multi_statement|not_select|"
                                        r"forbidden_keyword|incomplete_sql|unparseable_sql):", re.I)),
    (ErrorClass.AMBIGUOUS_COLUMN, re.compile(r"ambiguous column name", re.I)),
    (ErrorClass.NO_SUCH_COLUMN, re.compile(r"no such column", re.I)),
    (ErrorClass.NO_SUCH_TABLE, re.compile(r"no such table", re.I)),
    (ErrorClass.SYNTAX, re.compile(r"\bnear\b.*\bsyntax error\b|syntax error|"
                                   r"\bincomplete input\b", re.I)),
    (ErrorClass.TIMEOUT, re.compile(r"interrupted|exceeded .*s\b|timeout", re.I)),
    (ErrorClass.EMPTY_RESULT, re.compile(r"empty result|zero rows", re.I)),
]


def classify(error_class: str | None, error_message: str | None, is_empty: bool = False) -> ErrorClass:
    """把沙箱返回的 ``error_class`` / ``error_message`` 归一成 :class:`ErrorClass`。

    Parameters
    ----------
    error_class:
        沙箱已经给出的一级分类（如 ``"db_error"``）；为 None 表示执行成功。
    error_message:
        原始报错文本。
    is_empty:
        执行成功但零行时为 True。
    """
    if error_class is None:
        return ErrorClass.EMPTY_RESULT if is_empty else ErrorClass.UNKNOWN

    # 沙箱已经确定的类别直接透传
    try:
        known = ErrorClass(error_class)
        if known is not ErrorClass.DB_ERROR:
            return known
    except ValueError:
        pass

    msg = error_message or ""
    for cls, pat in _PATTERNS:
        if pat.search(msg):
            return cls
    return ErrorClass.DB_ERROR if error_class == "db_error" else ErrorClass.UNKNOWN


def extract_offender(message: str) -> str | None:
    """从报错里抠出"惹事的标识符"，用来在反馈里点名。

    ``no such column: T2.emp_name`` → ``T2.emp_name``
    ``no such table: employees``    → ``employees``
    ``ambiguous column name: id``   → ``id``
    """
    m = re.search(r"(?:no such column|no such table|ambiguous column name):\s*([^\s;]+)", message, re.I)
    return m.group(1) if m else None


@dataclass(slots=True)
class ClassifiedError:
    """分类结果 + 供反馈模板使用的结构化字段。"""

    category: ErrorClass
    raw_message: str
    offender: str | None = None

    @property
    def is_schema_error(self) -> bool:
        """schema 类错误 = 模型没看清表结构，这类最容易被反馈修好。"""
        return self.category in {
            ErrorClass.NO_SUCH_COLUMN,
            ErrorClass.NO_SUCH_TABLE,
            ErrorClass.AMBIGUOUS_COLUMN,
        }


def classify_result(success: bool, error_class: str | None, error_message: str,
                    is_empty: bool) -> ClassifiedError:
    """对 :class:`envs.sql_sandbox.QueryResult` 的字段做一次分类。"""
    if success and not is_empty:
        return ClassifiedError(ErrorClass.UNKNOWN, "")
    cat = classify(error_class, error_message, is_empty=is_empty)
    offender = extract_offender(error_message or "") if cat in {
        ErrorClass.NO_SUCH_COLUMN, ErrorClass.NO_SUCH_TABLE, ErrorClass.AMBIGUOUS_COLUMN
    } else None
    return ClassifiedError(cat, error_message or "", offender)
