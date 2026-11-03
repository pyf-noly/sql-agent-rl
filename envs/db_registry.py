"""数据库注册表：读 schema、缓存、按库名定位 sqlite 文件。

Spider 1.0 的目录约定（下载后长这样）::

    data/raw/spider/database/<db_id>/<db_id>.sqlite
    data/raw/spider/train_spider.json
    data/raw/spider/dev.json
    data/raw/spider/tables.json

这里只依赖 ``database/<db_id>/<db_id>.sqlite`` 这一条约定，
所以换成 BIRD（它也是 sqlite）时只要改 ``db_root``。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from envs.sql_sandbox import open_readonly


@dataclass(frozen=True, slots=True)
class TableInfo:
    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...] = ()

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

    def render_ddl(self, max_cols: int = 60) -> str:
        """渲染成 ``CREATE TABLE`` 风格的 schema 文本，直接塞进 prompt。

        为什么用 DDL 而不是 JSON：模型对 DDL 的先验更强，且 token 更省。
        """
        lines: list[str] = []
        budget = max_cols
        for t in self.tables:
            cols = []
            for c in t.columns:
                if budget <= 0:
                    break
                cols.append(c)
                budget -= 1
            suffix = "" if len(cols) == len(t.columns) else ", ..."
            pk = f"  -- PK: {', '.join(t.primary_key)}" if t.primary_key else ""
            lines.append(f"CREATE TABLE {t.name} ({', '.join(cols)}{suffix});{pk}")
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
            pk = [r[1] for r in info if r[5]]
            tables.append(TableInfo(name=tname, columns=tuple(cols), primary_key=tuple(pk)))
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
