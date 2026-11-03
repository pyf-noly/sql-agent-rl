"""数据库注册表测试：定位、schema 读取、DDL 渲染、缓存。"""

from __future__ import annotations

from pathlib import Path

import pytest

from envs.db_registry import DatabaseRegistry, DbSchema, TableInfo


class TestRegistry:
    def test_db_path_layout(self, db_root: Path) -> None:
        reg = DatabaseRegistry(db_root)
        p = reg.db_path("company_1")
        assert p.name == "company_1.sqlite"
        assert p.is_file()

    def test_missing_db_raises(self, db_root: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatabaseRegistry(db_root).db_path("not_a_db")

    def test_list_db_ids(self, db_root: Path) -> None:
        assert DatabaseRegistry(db_root).list_db_ids() == ["company_1"]

    def test_schema_tables(self, db_root: Path) -> None:
        s = DatabaseRegistry(db_root).schema("company_1")
        assert isinstance(s, DbSchema)
        assert s.db_id == "company_1"
        assert s.table_names == ["department", "employee"]

    def test_schema_columns_and_pk(self, db_root: Path) -> None:
        s = DatabaseRegistry(db_root).schema("company_1")
        emp = next(t for t in s.tables if t.name == "employee")
        assert emp.columns == ("emp_id", "emp_name", "dept_id", "salary")
        assert emp.primary_key == ("emp_id",)
        assert emp.has_primary_key

    def test_schema_is_cached(self, db_root: Path) -> None:
        reg = DatabaseRegistry(db_root)
        a = reg.schema("company_1")
        b = reg.schema("company_1")
        assert a is b  # lru_cache 返回同一对象

    def test_columns_by_table(self, db_root: Path) -> None:
        s = DatabaseRegistry(db_root).schema("company_1")
        cols = s.columns_by_table
        assert cols["department"] == ["dept_id", "dept_name", "budget"]
        assert list(cols) == ["department", "employee"]

    def test_render_ddl(self, db_root: Path) -> None:
        s = DatabaseRegistry(db_root).schema("company_1")
        ddl = s.render_ddl()
        assert "CREATE TABLE department" in ddl
        assert "dept_name" in ddl
        assert "PK: emp_id" in ddl

    def test_render_ddl_respects_budget(self, db_root: Path) -> None:
        s = DatabaseRegistry(db_root).schema("company_1")
        ddl = s.render_ddl(max_cols=3)
        # 7 个列挤在 3 个预算内，应该出现省略号
        assert "..." in ddl

    def test_tableinfo_immutable(self) -> None:
        t = TableInfo(name="t", columns=("a",))
        with pytest.raises(Exception):
            t.name = "x"  # frozen dataclass
