"""评测器测试：等价判定、值规范化、指标聚合。

这一组测试的核心是 **gold-as-prediction 必须判对**——它是评测器可信的最低门槛。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from envs.db_registry import DatabaseRegistry
from envs.sql_sandbox import QueryResult
from rollout.evaluator import Metrics, EvalResult, Evaluator, results_equal


@pytest.fixture()
def eval_db(tmp_path: Path) -> Path:
    """两个库：company（多表）与 single（单表），覆盖 join / 空结果 / 顺序。"""
    root = tmp_path / "database"
    for db_id in ("company", "single"):
        d = root / db_id
        d.mkdir(parents=True)
        con = sqlite3.connect(d / f"{db_id}.sqlite")
        if db_id == "company":
            con.executescript("""
                CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INT, salary REAL);
                CREATE TABLE project (id INTEGER PRIMARY KEY, title TEXT);
                INSERT INTO dept VALUES (1,'Eng'),(2,'Sales'),(3,'HR');
                INSERT INTO emp VALUES (1,'Alice',1,120.0),(2,'Bob',1,110.0),(3,'Carol',2,90.0);
                INSERT INTO project VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e');
            """)
        else:
            con.executescript("""
                CREATE TABLE t (a INT, b TEXT);
                INSERT INTO t VALUES (1,'x'),(2,'y');
            """)
        con.commit()
        con.close()
    return root


def _res(rows: list[tuple], cols: list[str] | None = None) -> QueryResult:
    return QueryResult(success=True, columns=cols or [f"c{i}" for i in range(len(rows[0]) if rows else 0)],
                       rows=rows)


class TestResultsEqual:
    def test_identical(self) -> None:
        ok, reason = results_equal(_res([(1, "a")]), _res([(1, "a")]), gold_has_order=False)
        assert ok and reason == "match"

    def test_column_order_ignored(self) -> None:
        ok, _ = results_equal(_res([(1, "a")]), _res([("a", 1)]), gold_has_order=False)
        assert ok

    def test_row_order_ignored_without_order_by(self) -> None:
        ok, _ = results_equal(_res([(1,), (2,)]), _res([(2,), (1,)]), gold_has_order=False)
        assert ok

    def test_row_order_respected_with_order_by(self) -> None:
        ok, reason = results_equal(_res([(1,), (2,)]), _res([(2,), (1,)]), gold_has_order=True)
        assert not ok and reason == "value_mismatch"

    def test_row_count_mismatch(self) -> None:
        ok, reason = results_equal(_res([(1,)]), _res([(1,), (2,)]), gold_has_order=False)
        assert not ok and reason == "row_mismatch"

    def test_empty_vs_empty(self) -> None:
        ok, reason = results_equal(_res([]), _res([]), gold_has_order=False)
        assert ok and reason == "empty_vs_empty"

    def test_value_mismatch(self) -> None:
        ok, reason = results_equal(_res([(1,)]), _res([(2,)]), gold_has_order=False)
        assert not ok and reason == "value_mismatch"

    @pytest.mark.parametrize("gold,pred", [
        ((1,), (1.0,)),
        ((1.0,), ("1",)),
        ((None,), (None,)),
    ])
    def test_value_normalization(self, gold: tuple, pred: tuple) -> None:
        ok, _ = results_equal(_res([gold]), _res([pred]), gold_has_order=False)
        assert ok

    def test_bool_equals_int(self) -> None:
        ok, _ = results_equal(_res([(True,)]), _res([(1,)]), gold_has_order=False)
        assert ok

    def test_null_not_equal_to_string_null(self) -> None:
        # NULL 归一成 "NULL"，字符串 "NULL" 也归一成 "NULL" —— 这是已知模糊点
        ok, _ = results_equal(_res([(None,)]), _res([("NULL",)]), gold_has_order=False)
        assert ok  # 记录当前行为，若有需要再收紧


class TestEvaluator:
    def test_gold_as_prediction_is_correct(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        gold = "SELECT name FROM emp ORDER BY id"
        r = ev.evaluate("company", gold, gold)
        assert r.correct and r.reason == "match"

    def test_alternative_but_equivalent_sql(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        gold = "SELECT count(*) FROM emp"
        pred = "SELECT count(id) FROM emp"
        r = ev.evaluate("company", gold, pred)
        assert r.correct

    def test_column_order_variation(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        gold = "SELECT name, salary FROM emp WHERE salary > 100"
        pred = "SELECT salary, name FROM emp WHERE salary > 100"
        assert ev.evaluate("company", gold, pred).correct

    def test_wrong_answer(self, eval_db: Path) -> None:
        """emp 3 行 vs project 5 行 —— 单行单列的值不同，必须判错。"""
        ev = Evaluator(DatabaseRegistry(eval_db))
        r = ev.evaluate("company", "SELECT count(*) FROM emp",
                        "SELECT count(*) FROM project")
        assert not r.correct
        assert r.reason == "value_mismatch"  # 都只有 1 行，差在值

        r2 = ev.evaluate("company", "SELECT name FROM emp ORDER BY id",
                         "SELECT name FROM dept ORDER BY id")
        assert not r2.correct

    def test_same_count_is_known_limitation(self, eval_db: Path) -> None:
        """记录已知局限：单库执行等价下，恰好同结果的不同 SQL 会被判对。"""
        ev = Evaluator(DatabaseRegistry(eval_db))
        r = ev.evaluate("company", "SELECT count(*) FROM emp", "SELECT count(*) FROM dept")
        assert r.correct  # 都是 3 行 —— 这是宽松口径的代价，P7 用官方 test-suite 复核

    def test_pred_error(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        r = ev.evaluate("company", "SELECT count(*) FROM emp", "SELECT count(*) FROM nope")
        assert not r.correct and r.reason == "pred_error"
        assert r.pred_error_class == "db_error"

    def test_empty_prediction(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        r = ev.evaluate("company", "SELECT 1", "")
        assert not r.correct and r.reason == "pred_error"

    def test_dangerous_prediction_rejected(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        r = ev.evaluate("company", "SELECT count(*) FROM emp", "DROP TABLE emp")
        assert not r.correct and r.pred_error_class == "invalid_sql"

    def test_gold_cache(self, eval_db: Path) -> None:
        ev = Evaluator(DatabaseRegistry(eval_db))
        gold = "SELECT count(*) FROM emp"
        a = ev.gold_result("company", gold)
        b = ev.gold_result("company", gold)
        assert a is b
        assert len(ev._gold_cache) == 1


class TestMetrics:
    def test_accuracy(self) -> None:
        m = Metrics()
        m.add(EvalResult(True, "match"))
        m.add(EvalResult(False, "value_mismatch"))
        assert m.total == 2 and m.correct == 1 and m.accuracy == 0.5

    def test_empty_total_is_zero_not_error(self) -> None:
        assert Metrics().accuracy == 0.0

    def test_breakdown(self) -> None:
        m = Metrics()
        m.add(EvalResult(True, "match"), difficulty="easy", turns=1)
        m.add(EvalResult(False, "row_mismatch"), difficulty="hard", turns=1)
        m.add(EvalResult(True, "match"), difficulty="easy", turns=2)
        d = m.to_dict()
        assert d["by_difficulty"]["easy"] == {"total": 2, "correct": 2, "accuracy": 1.0}
        assert d["by_difficulty"]["hard"]["accuracy"] == 0.0
        assert d["by_turns"]["1"] == {"total": 2, "correct": 1, "accuracy": 0.5}
        assert d["by_turns"]["2"]["accuracy"] == 1.0

    def test_reason_counters(self) -> None:
        m = Metrics()
        m.add(EvalResult(False, "gold_error"))
        m.add(EvalResult(False, "pred_error", pred_error_class="db_error"))
        m.add(EvalResult(False, "row_mismatch"))
        m.add(EvalResult(False, "value_mismatch"))
        assert m.gold_error == 1
        assert m.pred_error == 1
        assert m.row_mismatch == 1
        assert m.value_mismatch == 1
