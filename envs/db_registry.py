"""数据库注册表：读 schema、缓存、按库名定位 sqlite 文件。

Spider 1.0 的目录约定（下载后长这样）::

    data/raw/spider/database/<db_id>/<db_id>.sqlite
    data/raw/spider/train_spider.json
    data/raw/spider/dev.json
    data/raw/spider/tables.json

BIRD 的目录约定（``dev.zip`` 解压后）::

    dev_20240627/dev_databases/<db_id>/<db_id>.sqlite
    dev_20240627/dev.json
    dev_20240627/dev_tables.json

两者都是「``<root>/<db_id>/<db_id>.sqlite``」布局，所以只要换 ``db_root``
就能复用；差异只在 JSON 字段名，用 :func:`load_bird_examples` 抹平。
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from envs.sql_sandbox import open_readonly

#: 需要加引号的标识符：含空格、括号、连字符等非 ``[A-Za-z0-9_]`` 字符。
_NEEDS_QUOTE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True, slots=True)
class TableInfo:
    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    """与 ``columns`` 一一对应的声明类型（来自 ``PRAGMA table_info``）。

    BIRD 的列名经常是缩写（``T1``/``T2``），不看类型根本猜不出含义，所以
    渲染 DDL 时可以选择带上它（Spider 默认不带，保持 P2~P5 的 prompt 不变）。
    """
    foreign_keys: tuple[tuple[str, str, str], ...] = ()
    """``(本表列, 引用表, 引用列)``。BIRD 的库都显式声明了外键，是 join 的关键线索。"""

    @property
    def has_primary_key(self) -> bool:
        return bool(self.primary_key)


@dataclass(frozen=True, slots=True)
class DbSchema:
    db_id: str
    path: Path
    tables: tuple[TableInfo, ...] = field(default_factory=tuple)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    @property
    def columns_by_table(self) -> dict[str, list[str]]:
        return {t.name: list(t.columns) for t in self.tables}

    def render_ddl(self, max_cols: int = 60, *, include_types: bool = False,
                   include_fk: bool = False, quote_identifiers: bool = False) -> str:
        """渲染成 ``CREATE TABLE`` 风格的 schema 文本，直接塞进 prompt。

        为什么用 DDL 而不是 JSON：模型对 DDL 的先验更强，且 token 更省。

        Parameters
        ----------
        max_cols:
            全局列预算（所有表共享）。默认 60 是 P2~P5 Spider 实验的配置；
            BIRD 的库列多（最大 199 列），要显式放大到 400，否则大部分列会被吞掉。
        include_types:
            是否写列类型。Spider 的类型信息不可靠（官方 tables.json 才有），
            所以默认关；BIRD 从 sqlite 读出来的类型可信，打开有实际收益。
        include_fk:
            是否把外键写成 ``-- FK: a -> b(c)`` 注释行。
        quote_identifiers:
            名字含空格/括号/连字符时是否加双引号。BIRD 大量列名是
            ``Free Meal Count (K-12)`` 这种，不引用的话模型会自己把空格删掉
            （实测写成 ``f.FreeMealCountK12``，直接执行报错）。Spider 的列名
            都是裸标识符，默认关可保持旧 prompt 逐字不变。
        """
        lines: list[str] = []

        def q(name: str) -> str:
            if quote_identifiers and _NEEDS_QUOTE.search(name):
                return '"' + name.replace('"', '""') + '"'
            return name

        budget = max_cols
        for t in self.tables:
            cols = []
            for i, c in enumerate(t.columns):
                if budget <= 0:
                    break
                ty = t.types[i] if include_types and i < len(t.types) else ""
                cols.append(f"{q(c)} {ty}".strip() if ty else q(c))
                budget -= 1
            suffix = "" if len(cols) == len(t.columns) else ", ..."
            pk = f"  -- PK: {', '.join(q(c) for c in t.primary_key)}" if t.primary_key else ""
            lines.append(f"CREATE TABLE {q(t.name)} ({', '.join(cols)}{suffix});{pk}")
            if include_fk:
                for from_col, ref_table, ref_col in t.foreign_keys:
                    lines.append(
                        f"-- FK: {t.name}.{from_col} -> {ref_table}.{ref_col}")
        return "\n".join(lines)


class DatabaseRegistry:
    """按 ``db_id`` 找 sqlite 文件并读 schema，结果带缓存。"""

    def __init__(self, db_root: str | Path):
        self.db_root = Path(db_root)

    def db_path(self, db_id: str) -> Path:
        # 兼容两种常见布局：<root>/<db_id>/<db_id>.sqlite 与 <root>/<db_id>.sqlite
        p1 = self.db_root / db_id / f"{db_id}.sqlite"
        if p1.is_file():
            return p1
        p2 = self.db_root / f"{db_id}.sqlite"
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"找不到数据库 {db_id}（在 {self.db_root} 下）")

    def schema(self, db_id: str) -> DbSchema:
        return _load_schema(str(self.db_path(db_id).resolve()))

    def list_db_ids(self) -> list[str]:
        if not self.db_root.is_dir():
            return []
        return sorted(
            p.name for p in self.db_root.iterdir()
            if p.is_dir() and (p / f"{p.name}.sqlite").is_file()
        )


@lru_cache(maxsize=512)
def _load_schema(path_str: str) -> DbSchema:
    path = Path(path_str)
    con = open_readonly(path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: list[TableInfo] = []
        for (tname,) in rows:
            # PRAGMA 的表名不能参数化，用双引号转义（表名来自 sqlite_master，可信）
            safe = tname.replace('"', '""')
            info = con.execute(f'PRAGMA table_info("{safe}")').fetchall()
            cols = [r[1] for r in info]
            types = [(r[2] or "") for r in info]
            pk = [r[1] for r in info if r[5]]
            # foreign_key_list 的列顺序：(id, seq, table, from, to, on_update, on_delete, match)
            fks = tuple(
                (r[3], r[2], r[4]) for r in
                con.execute(f'PRAGMA foreign_key_list("{safe}")').fetchall()
                if r[3] and r[2] and r[4]
            )
            tables.append(TableInfo(
                name=tname, columns=tuple(cols), primary_key=tuple(pk),
                types=tuple(types), foreign_keys=fks))
        # 优先用「文件名即 db_id」；兼容 <root>/<db_id>/<db_id>.sqlite 布局
        db_id = path.stem
        if path.parent.name == db_id:
            db_id = path.parent.name
        return DbSchema(db_id=db_id, path=path, tables=tuple(tables))
    finally:
        con.close()


def load_spider_examples(split_json: str | Path) -> list[dict]:
    """读 Spider 的 ``train_spider.json`` / ``dev.json``。

    每条样本形如::

        {"db_id": "department_management", "question": "...", "query": "SELECT ..."}
    """
    with open(split_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{split_json} 不是 Spider 的 JSON 列表格式")
    return data


def load_spider_tables(tables_json: str | Path) -> dict[str, dict]:
    """读 ``tables.json``，返回 ``{db_id: {...}}``。

    注意：这里**不用**它来构造 schema，而是用来做交叉校验
    （我们直接从 sqlite 文件读 schema，比 tables.json 更可信）。
    """
    with open(tables_json, encoding="utf-8") as f:
        data = json.load(f)
    return {item["db_id"]: item for item in data}


#: BIRD 的三档难度 → 本项目的四档（Spider 才有 extra hard）。
#: 不映射就会在多处 ``by_difficulty`` 统计里出现两套词，报表没法并排看。
BIRD_DIFFICULTY = {"simple": "easy", "moderate": "medium", "challenging": "hard"}


def load_bird_examples(split_json: str | Path) -> list[dict]:
    """读 BIRD 的 ``dev.json`` / ``train.json``，归一成本项目内部字段。

    BIRD 的原始字段与 Spider 不同（``SQL`` 大写、多一个 ``evidence``、还有
    ``question_id`` 与官方难度），这里一次性映射掉，下游（runner / 数据管线）
    只看见和 Spider 一样的 ``db_id/question/query/difficulty``：

    ============  ===================================
    BIRD 字段      本项目字段
    ============  ===================================
    question       question
    SQL            query
    evidence       evidence（空字符串表示没有外部知识）
    difficulty     difficulty（simple/moderate/challenging → easy/medium/hard）
    question_id    question_id
    ============  ===================================

    注意 ``evidence`` 是**外部知识提示**（例如"T2 表示用户评分"），不是答案；
    拼进 prompt 是 BIRD 的官方设定，不算泄漏。
    """
    with open(split_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{split_json} 不是 BIRD 的 JSON 列表格式")

    out: list[dict] = []
    for rec in data:
        sql = rec.get("SQL") or rec.get("query") or ""
        if not sql:
            raise KeyError(f"BIRD 样本缺 SQL 字段: {rec.get('question_id')}")
        out.append({
            "question_id": rec.get("question_id"),
            "db_id": rec["db_id"],
            "question": rec["question"],
            "evidence": (rec.get("evidence") or "").strip(),
            "query": sql,
            "difficulty": BIRD_DIFFICULTY.get(rec.get("difficulty", ""), rec.get("difficulty")),
            "difficulty_raw": rec.get("difficulty"),
        })
    return out
