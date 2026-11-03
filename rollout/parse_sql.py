"""从模型输出里抽 SQL 代码块。

模型不会永远按格式输出：可能套 ```sql、```sqlite、```（无语言），也可能裸写 SQL，
还可能先来一句 "Here is the query:"。解析器要**宽容**，但要能报告"到底抽到了什么"，
否则训练时你分不清"模型答错了"和"解析器没抽出来"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: 带语言标记的代码块：```sql ... ``` / ```sqlite ... ``` / ```SQL ...
_FENCED_RE = re.compile(r"```[ \t]*(sql|sqlite|SQL|SQLite)?[ \t]*\r?\n(.*?)```", re.S)

#: 没有任何代码块时，退而求其次：从文本里抓第一个 SELECT/WITH 开头到结尾
_BARE_RE = re.compile(r"((?:WITH|SELECT)\b.*?)(?:;|$)", re.S | re.I)


@dataclass(slots=True)
class ParsedSql:
    sql: str
    """抽出的 SQL（已去分号、去首尾空白）；抽不到为空串。"""
    source: str
    """来源：``fence`` / ``bare`` / ``none``。"""
    raw: str = ""
    """模型原始输出，便于失败分析。"""

    @property
    def ok(self) -> bool:
        return bool(self.sql)


def _clean(sql: str) -> str:
    s = sql.strip()
    # 去掉包裹的分号（多个）
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def parse_sql(text: str | None) -> ParsedSql:
    """从模型输出里解析 SQL。永不抛异常。"""
    raw = text or ""
    if not raw.strip():
        return ParsedSql("", "none", raw)

    # 1) 优先取带 sql 语言标记的块
    blocks = _FENCED_RE.findall(raw)
    if blocks:
        # 有多个块时取最后一个含 SELECT 的（模型常先解释、最后给答案）
        picked = None
        for lang, body in blocks:
            if re.search(r"\b(select|with)\b", body, re.I):
                picked = body
        if picked is not None:
            return ParsedSql(_clean(picked), "fence", raw)
        # 有块但里面没有 SELECT —— 也算抽到，让后续校验去拒绝
        return ParsedSql(_clean(blocks[-1][1]), "fence", raw)

    # 2) 没有代码块：裸文本里抓 SELECT/WITH
    m = _BARE_RE.search(raw)
    if m:
        return ParsedSql(_clean(m.group(1)), "bare", raw)

    return ParsedSql("", "none", raw)


def extract_sql(text: str | None) -> str:
    """便捷函数：只要 SQL 字符串。"""
    return parse_sql(text).sql
