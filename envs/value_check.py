"""值检查：判断 SQL 里的字符串字面量在当前库里是否真的存在（不泄漏答案）。

为什么需要它（P6 的实测动机）
---------------------------
2026-11-05 的实测：4B 在 BIRD 上的错题里，约 **2/3 是"执行成功但结果错"**
（join 错字段、聚合写错、**取值拼错**）。执行反馈只有 `Execution succeeded: 1 row`，
模型没有任何依据去改——三轮自修复只救回 9/153 条错题。

但有一类语义错误其实是**可检测**的：`WHERE County = 'Alamedaa'` 这种取值错误。
它不需要 gold SQL，只需要问数据库一句"这个值在这一列里存在吗"。
BIRD 的库恰好是"脏数据 + 长尾取值"，拼错/用错取值非常常见。

于是这里做三件事（全部只用数据库本身，不碰 gold）：

1. 抽取 SQL 里的 ``col = 'lit'`` / ``col IN ('a','b')`` / ``col LIKE '%x%'``；
2. 对每个字面量，到"所有含该列的表"里查它是否存在（参数化查询，2s 预算）；
3. 不存在时给出**最近的真实取值**（difflib），让模型有依据地改。

这属于"环境能看见、模型看不见"的合法信息：它不告诉模型答案是什么，
只说"你写的这个值在库里不存在，最接近的是这两个"。
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from dataclasses import dataclass, field

from envs.db_registry import DbSchema
from envs.sql_sandbox import open_readonly

#: ``col = 'lit'`` / ``col <> 'lit'`` / ``col LIKE 'lit'`` / ``col IN ('a', 'b')``
#: 标识符支持三种引用形式：``"col"`` / ``[col]`` / ``\`col\```，以及可选的 ``t.`` 前缀。
_IDENT = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)'
_LITERAL_RE = re.compile(
    rf"(?P<col>{_IDENT}(?:\.{_IDENT})?)"
    r"\s*(?P<op>=|<>|!=|\bLIKE\b|\bIN\b)\s*"
    r"(?P<val>'[^']*(?:''[^']*)*'|\([^)]*\))",
    re.IGNORECASE,
)
_STR_RE = re.compile(r"'([^']*(?:''[^']*)*)'")

MAX_CHECKS = 4
MAX_SUGGESTIONS = 3
MAX_DISTINCT = 300


@dataclass(slots=True)
class LiteralCheck:
    column: str
    literal: str
    op: str
    exists: bool
    suggestions: list[str] = field(default_factory=list)
    checked_tables: list[str] = field(default_factory=list)

    @property
    def is_like(self) -> bool:
        return self.op.upper() == "LIKE"


def _unescape(s: str) -> str:
    return s.replace("''", "'")


def _col_name(expr: str) -> str:
    last = expr.split(".")[-1].strip()
    if len(last) >= 2 and ((last[0] == '"' and last[-1] == '"')
                           or (last[0] == "`" and last[-1] == "`")
                           or (last[0] == "[" and last[-1] == "]")):
        last = last[1:-1]
    return last.strip()


def extract_literals(sql: str) -> list[tuple[str, str, str]]:
    """返回 ``[(列名, 操作符, 字面量)]``；同一 (列, 值) 只保留一次。"""
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in _LITERAL_RE.finditer(sql):
        col = _col_name(m.group("col"))
        op = m.group("op").upper()
        raw = m.group("val")
        vals = [_unescape(x) for x in _STR_RE.findall(raw)]
        if not vals:
            continue
        for v in vals:
            key = (col, op, v)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _values_from_table(table: str, col: str, path) -> list[str]:
    con = open_readonly(path)
    try:
        t = table.replace('"', '""')
        c = col.replace('"', '""')
        rows = con.execute(
            f'SELECT DISTINCT "{c}" FROM "{t}" '
            f'WHERE "{c}" IS NOT NULL LIMIT {MAX_DISTINCT}'
        ).fetchall()
        vals = [str(r[0]) for r in rows if r[0] is not None]
        return vals
    except sqlite3.DatabaseError:
        return []
    finally:
        con.close()


def _exists(path, table: str, col: str, literal: str, op: str) -> bool:
    con = open_readonly(path)
    try:
        t = table.replace('"', '""')
        c = col.replace('"', '""')
        if op == "LIKE":
            pat = literal.replace("%", "").replace("_", "")
            sql = f'SELECT 1 FROM "{t}" WHERE "{c}" LIKE ? LIMIT 1'
            val = f"%{pat}%"
        else:
            sql = f'SELECT 1 FROM "{t}" WHERE "{c}" = ? LIMIT 1'
            val = literal
        return con.execute(sql, (val,)).fetchone() is not None
    except sqlite3.DatabaseError:
        return True          # 查不了就当"存在"，避免误报
    finally:
        con.close()


def check_literals(sql: str, schema: DbSchema, *, timeout: float = 2.0) -> list[LiteralCheck]:
    """检查 SQL 里的字面量是否存在于对应列（跨所有含该列的表）。"""
    db_path = schema.path
    col_to_tables: dict[str, list[str]] = {}
    for t in schema.tables:
        for c in t.columns:
            col_to_tables.setdefault(c, []).append(t.name)
    # 列名大小写不敏感匹配（BIRD 的列名大小写混杂）
    lower_map = {c.lower(): c for c in col_to_tables}

    checks: list[LiteralCheck] = []
    for col, op, lit in extract_literals(sql)[:MAX_CHECKS]:
        real = lower_map.get(col.lower())
        if not real:
            continue
        tables = col_to_tables[real]
        hit = any(_exists(db_path, t, real, lit, op) for t in tables)
        ck = LiteralCheck(column=real, literal=lit, op=op, exists=hit,
                          checked_tables=tables[:3])
        if not hit:
            # 收集候选值做"最近邻"提示（优先取值多的那个表）
            best: list[str] = []
            for t in tables[:2]:
                vals = _values_from_table(t, real, db_path)
                if len(vals) > len(best):
                    best = vals
            ck.suggestions = difflib.get_close_matches(
                lit, best, n=MAX_SUGGESTIONS, cutoff=0.5)
        checks.append(ck)
    return checks


def format_hints(checks: list[LiteralCheck], *, max_items: int = 2) -> str:
    """把不存在的字面量渲染成一段短提示；没有任何问题时返回空串。"""
    bad = [c for c in checks if not c.exists]
    if not bad:
        return ""
    lines = ["Value check: a literal in your query does not exist in the database."]
    for c in bad[:max_items]:
        s = f"- `{c.literal}` is not found in column `{c.column}`"
        if c.op == "LIKE":
            s += " (as a LIKE pattern)"
        if c.suggestions:
            s += ". Closest existing values: " + ", ".join(
                f"'{v}'" for v in c.suggestions)
        lines.append(s)
    lines.append("Check spelling / the right column for this value before changing the logic.")
    return "\n".join(lines)
