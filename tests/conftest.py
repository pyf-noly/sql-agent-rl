"""pytest 共享夹具：造一个小型 sqlite 库，供所有测试用。

为什么不直接下 Spider？因为单测要**快且离线**。Spider 留给集成测试（P2）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_db(tmp_path_factory) -> Path:
    """一个两表 + 外键 + 含 NULL 的小库，覆盖大部分 SQL 场景。"""
    root = tmp_path_factory.mktemp("dbs")
    db_dir = root / "company_1"
    db_dir.mkdir()
    path = db_dir / "company_1.sqlite"

    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE department (
            dept_id   INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL,
            budget    REAL
        );
        CREATE TABLE employee (
            emp_id   INTEGER PRIMARY KEY,
            emp_name TEXT NOT NULL,
            dept_id  INTEGER,
            salary   REAL,
            FOREIGN KEY (dept_id) REFERENCES department(dept_id)
        );
        INSERT INTO department VALUES (1, 'Engineering', 1000.0);
        INSERT INTO department VALUES (2, 'Sales', 500.0);
        INSERT INTO department VALUES (3, 'Empty Dept', NULL);
        INSERT INTO employee VALUES (10, 'Alice', 1, 120.0);
        INSERT INTO employee VALUES (11, 'Bob',   1, 110.0);
        INSERT INTO employee VALUES (12, 'Carol', 2,  90.0);
        INSERT INTO employee VALUES (13, 'Dave',  NULL, 80.0);
        """
    )
    con.commit()
    con.close()
    return path


@pytest.fixture(scope="session")
def db_root(tiny_db: Path) -> Path:
    """``<root>/<db_id>/<db_id>.sqlite`` 布局。"""
    return tiny_db.parent.parent
