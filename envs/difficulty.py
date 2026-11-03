"""按 SQL 结构给样本打难度标签（Spider 论文四档的工程近似）。

为什么要自己打标签
------------------
公开的 HuggingFace 镜像把 ``difficulty`` 字段丢了，只剩 db_id/question/query。
分层抽样必须有一个稳定的分层依据，所以这里按 **SQL 结构** 复刻 Spider 论文的四档：

- EASY   ：单表，无嵌套，最多一个聚合
- MEDIUM ：多表 JOIN，或单表 + 嵌套/聚合
- HARD   ：多表 JOIN + 聚合/嵌套/集合运算
- EXTRA  ：两层以上嵌套，或带 EXCEPT/INTERSECT/UNION，或嵌套 + 多表

这不是"真值"，是给抽样和指标分桶用的**工程标签**。它的好处是：
1. 完全确定性，可复现；
2. 不依赖任何外部标注，换数据集也能用；
3. 分桶后能看出"RL 的提升集中在哪一档"——这正是 P6 要的机制解释。

用法::

    from envs.difficulty import classify_query
    classify_query("SELECT count(*) FROM singer")        # -> "easy"
    classify_query("SELECT ... FROM a JOIN b ON ...")    # -> "medium"
"""

from __future__ import annotations

import re
from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTRA = "extra"


#: 聚合函数（注意 ``count`` 常被写成 ``COUNT``）
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.I)
_SET_OP_RE = re.compile(r"\b(except|intersect|union)\b", re.I)
_JOIN_RE = re.compile(r"\bjoin\b", re.I)
_FROM_RE = re.compile(r"\bfrom\b", re.I)
_GROUP_RE = re.compile(r"\bgroup\s+by\b", re.I)
_HAVING_RE = re.compile(r"\bhaving\b", re.I)
_ORDER_RE = re.compile(r"\border\s+by\b", re.I)


def _strip_strings(sql: str) -> str:
    """把字符串字面量换成占位符，避免里面的关键字被误数。"""
    return re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)


def _count_subqueries(sql: str) -> int:
    """数 ``(SELECT ...)`` 的层数（忽略括号在字符串里的情况）。"""
    s = _strip_strings(sql)
    depth = 0
    count = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        else:
            continue
    # 更精确：按 "( select" 出现次数算子查询个数
    count = len(re.findall(r"\(\s*select\b", s, re.I))
    return count


def _max_nesting(sql: str) -> int:
    """计算 ``(select ...)`` 的最大嵌套深度。"""
    s = _strip_strings(sql).lower()
    stack = 0
    best = 0
    for m in re.finditer(r"\(|\bselect\b", s):
        tok = m.group(0)
        if tok == "(":
            stack += 1
        else:
            best = max(best, stack)
    return best


def _count_tables(sql: str) -> int:
    """粗估 FROM/JOIN 里引用的表数量（不含子查询里的）。"""
    s = _strip_strings(sql)
    # 去掉子查询体，避免把内层表算进来
    flat = re.sub(r"\([^()]*select[^()]*\)", " ", s, flags=re.I)
    names: set[str] = set()
    for m in re.finditer(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", flat, re.I):
        name = m.group(1).lower()
        if name in {"select", "where", "as"}:
            continue
        names.add(name)
    # 逗号连接的表：FROM a, b
    for m in re.finditer(r"\bfrom\s+([^;]+?)(?:\bwhere\b|\bgroup\b|\border\b|\bhaving\b|$)",
                         flat, re.I | re.S):
        for part in m.group(1).split(","):
            tok = part.strip().split()
            if tok and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok[0]):
                names.add(tok[0].lower())
    return len(names)


def classify_query(sql: str) -> Difficulty:
    """按结构复杂度给一条 SQL 打标签。"""
    if not sql or not sql.strip():
        return Difficulty.EASY

    s = _strip_strings(sql)
    n_sub = _count_subqueries(s)
    max_depth = _max_nesting(s)
    n_tables = _count_tables(s)
    has_agg = bool(_AGG_RE.search(s))
    has_set = bool(_SET_OP_RE.search(s))
    has_join = bool(_JOIN_RE.search(s)) or n_tables > 1
    has_group = bool(_GROUP_RE.search(s)) or bool(_HAVING_RE.search(s))

    # EXTRA：多层嵌套 / 集合运算 / 嵌套 + 多表
    if max_depth >= 2 or has_set or (n_sub >= 2) or (n_sub >= 1 and n_tables >= 2):
        return Difficulty.EXTRA

    # HARD：多表 + （聚合或分组或嵌套）
    if has_join and (has_agg or has_group or n_sub >= 1):
        return Difficulty.HARD

    # MEDIUM：多表但简单，或单表但带聚合/嵌套/分组
    if has_join or has_agg or has_group or n_sub >= 1:
        return Difficulty.MEDIUM

    return Difficulty.EASY


def classify_many(rows: list[dict], key: str = "query") -> list[dict]:
    """给一批样本补上 ``difficulty`` 字段（不修改入参，返回新列表）。"""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["difficulty"] = classify_query(r.get(key, "")).value
        out.append(r2)
    return out
