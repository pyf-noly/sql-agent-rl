"""难度分层测试：锁住"什么结构算哪一档"的行为。

这些断言不是"真理"，是**约定**——写下来是为了将来改分类器时知道自己动了什么。
"""

from __future__ import annotations

import pytest

from envs.difficulty import Difficulty, classify_many, classify_query


class TestClassifyQuery:
    @pytest.mark.parametrize("sql", [
        "SELECT name FROM singer",
        "SELECT name, age FROM singer ORDER BY age DESC",
        "SELECT * FROM employee WHERE salary > 100",
        "SELECT DISTINCT country FROM singer",
    ])
    def test_simple_single_table_is_easy(self, sql: str) -> None:
        assert classify_query(sql) is Difficulty.EASY

    @pytest.mark.parametrize("sql", [
        "SELECT count(*) FROM singer",
        "SELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'",
        "SELECT country, count(*) FROM singer GROUP BY country",
        "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)",
    ])
    def test_single_table_with_agg_or_nested_is_medium(self, sql: str) -> None:
        assert classify_query(sql) is Difficulty.MEDIUM

    @pytest.mark.parametrize("sql", [
        # 多表 + 聚合/分组 → hard
        "SELECT T2.name, count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id",
        "SELECT count(*) FROM a, b WHERE a.id = b.a_id",
        "SELECT a.x, count(*) FROM a JOIN b ON a.id = b.a_id GROUP BY a.x",
    ])
    def test_multi_table_with_agg_is_hard(self, sql: str) -> None:
        assert classify_query(sql) is Difficulty.HARD

    @pytest.mark.parametrize("sql", [
        # 多表但没聚合/分组/嵌套 → 只是 medium（实现如此，测试按实现写）
        "SELECT a.x, b.y FROM a JOIN b ON a.id = b.a_id",
        "SELECT a.name FROM a JOIN b ON a.id = b.a_id WHERE b.flag = 1",
    ])
    def test_plain_join_is_medium(self, sql: str) -> None:
        assert classify_query(sql) is Difficulty.MEDIUM

    @pytest.mark.parametrize("sql", [
        "SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30",
        "SELECT name FROM stadium EXCEPT SELECT name FROM stadium JOIN concert ON 1=1",
        "SELECT x FROM a WHERE x IN (SELECT y FROM b WHERE y IN (SELECT z FROM c))",
        # 多表 + 嵌套 → extra（n_sub>=1 且 n_tables>=2）
        "SELECT * FROM a JOIN b ON a.id=b.a_id WHERE a.id IN (SELECT id FROM c)",
    ])
    def test_set_ops_or_deep_nesting_is_extra(self, sql: str) -> None:
        assert classify_query(sql) is Difficulty.EXTRA

    def test_keywords_inside_strings_are_ignored(self) -> None:
        # 字符串里的 'JOIN' / 'select' 不能影响判定
        sql = "SELECT note FROM t WHERE note = 'please join select union except'"
        assert classify_query(sql) is Difficulty.EASY

    def test_empty_is_easy_not_crash(self) -> None:
        assert classify_query("") is Difficulty.EASY
        assert classify_query("   ") is Difficulty.EASY
        assert classify_query(None) is Difficulty.EASY  # type: ignore[arg-type]

    def test_case_insensitive(self) -> None:
        assert classify_query("select COUNT(*) from singer") is Difficulty.MEDIUM
        # 纯 JOIN 无聚合 → medium
        assert classify_query("SELECT * FROM a JOIN b ON a.x=b.x") is Difficulty.MEDIUM
        # JOIN + 聚合 → hard
        assert classify_query("SELECT count(*) FROM a JOIN b ON a.x=b.x") is Difficulty.HARD


class TestClassifyMany:
    def test_adds_field_without_mutating_input(self) -> None:
        rows = [{"db_id": "d", "question": "q", "query": "SELECT 1"}]
        out = classify_many(rows)
        assert "difficulty" not in rows[0]
        assert out[0]["difficulty"] == "easy"
        assert out[0]["db_id"] == "d"

    def test_preserves_all_rows(self) -> None:
        rows = [
            {"query": "SELECT 1"},
            {"query": "SELECT count(*) FROM t"},
            {"query": "SELECT count(*) FROM a JOIN b ON 1=1"},
        ]
        out = classify_many(rows)
        assert len(out) == 3
        assert [r["difficulty"] for r in out] == ["easy", "medium", "hard"]
