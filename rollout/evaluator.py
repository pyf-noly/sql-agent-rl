"""执行等价判定（Execution Accuracy）+ 指标聚合。

判定规则（写清楚，方便复现和答辩）
----------------------------------
两条 SQL 等价 ⇔ 在同一个数据库上执行后，结果"相同"，其中：

1. **行数必须一致**；
2. **列顺序忽略**：把每一行按某种规范排序后比较（因为 `SELECT a, b` 和
   `SELECT b, a` 在语义上通常被认为等价，Spider 的 exec 口径也允许）；
3. **行顺序**：gold 里有 `ORDER BY` 时才比较顺序，否则无序比较；
4. **值规范化**：`int/float/str` 统一转成字符串比较（`1` == `1.0` == `"1"`），
   `None` 统一成 `"NULL"`，`True/False` → `"1"/"0"`（SQLite 的布尔就是整数）；
5. **超时 / 执行失败 一律判错**（不能让"跑不出结果"混成"结果对"）。

已知局限（写进 README，别装作没有）
------------------------------------
这是"单库执行等价"，不是 Spider 官方的 test-suite（那会在多个随机数据库上比较，
能抓到 `WHERE col = 'x'` 这类值敏感的错误）。所以它**偏宽松**：
对 gold 数据上恰好结果相同的错误 SQL 会判对。P2 先用它跑通，P7 前可换官方
test-suite 复核主表——那时再报一次差距。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from envs.db_registry import DatabaseRegistry
from envs.sql_sandbox import QueryResult, SqlSandbox

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.I)


@dataclass(slots=True)
class EvalResult:
    correct: bool
    reason: str
    """判定原因：``match`` / ``gold_error`` / ``pred_error`` / ``row_mismatch`` / ``value_mismatch`` / ``empty_vs_empty``。"""
    gold_rows: int = 0
    pred_rows: int = 0
    pred_error_class: str | None = None


@dataclass(slots=True)
class Metrics:
    total: int = 0
    correct: int = 0
    gold_error: int = 0
    pred_error: int = 0
    row_mismatch: int = 0
    value_mismatch: int = 0
    by_difficulty: dict[str, dict[str, int]] = field(default_factory=dict)
    by_turns: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def add(self, r: EvalResult, *, difficulty: str | None = None, turns: int | None = None) -> None:
        self.total += 1
        if r.correct:
            self.correct += 1

        if r.reason == "gold_error":
            self.gold_error += 1
        elif r.reason in ("pred_error", "empty_vs_empty") or r.pred_error_class:
            self.pred_error += 1
        elif r.reason == "row_mismatch":
            self.row_mismatch += 1
        elif r.reason == "value_mismatch":
            self.value_mismatch += 1

        if difficulty:
            b = self.by_difficulty.setdefault(difficulty, {"total": 0, "correct": 0})
            b["total"] += 1
            b["correct"] += int(r.correct)
        if turns:
            b = self.by_turns.setdefault(turns, {"total": 0, "correct": 0})
            b["total"] += 1
            b["correct"] += int(r.correct)

    def to_dict(self) -> dict:
        def _rate(b: dict[str, int]) -> float:
            return b["correct"] / b["total"] if b["total"] else 0.0

        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "gold_error": self.gold_error,
            "pred_error": self.pred_error,
            "row_mismatch": self.row_mismatch,
            "value_mismatch": self.value_mismatch,
            "by_difficulty": {
                k: {**v, "accuracy": round(_rate(v), 4)}
                for k, v in sorted(self.by_difficulty.items())
            },
            "by_turns": {
                str(k): {**v, "accuracy": round(_rate(v), 4)}
                for k, v in sorted(self.by_turns.items())
            },
        }


# ---------------- 值规范化 ----------------

def _norm_value(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NAN"
        if v.is_integer():
            return str(int(v))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, bytes):
        return v.hex()
    s = str(v).strip()
    # 数字字符串按数值归一，避免 "1" vs "1.0" 的假阴性
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return s


def _norm_rows(rows: list[tuple]) -> list[tuple[str, ...]]:
    return [tuple(_norm_value(v) for v in row) for row in rows]


def _canonical(rows: list[tuple]) -> list[tuple[str, ...]]:
    """每行内部排序（忽略列顺序），再整体排序（忽略行顺序）。"""
    normed = [tuple(sorted(r)) for r in _norm_rows(rows)]
    return sorted(normed)


def results_equal(gold: QueryResult, pred: QueryResult, *, gold_has_order: bool) -> tuple[bool, str]:
    """比较两个执行结果。返回 (是否等价, 原因)。"""
    if gold.row_count != pred.row_count:
        return False, "row_mismatch"

    if gold.row_count == 0:
        # 双方都空：可能都是"正确地没有匹配"，也可能都是错的。保守判对，
        # 但把原因单独记为 empty_vs_empty，方便事后审计这类样本占比。
        return True, "empty_vs_empty"

    g = _norm_rows(gold.rows)
    p = _norm_rows(pred.rows)

    if gold_has_order:
        # 比顺序：逐行比，但列顺序仍然忽略
        g2 = [tuple(sorted(r)) for r in g]
        p2 = [tuple(sorted(r)) for r in p]
        return (g2 == p2), ("match" if g2 == p2 else "value_mismatch")

    g3 = _canonical(gold.rows)
    p3 = _canonical(pred.rows)
    return (g3 == p3), ("match" if g3 == p3 else "value_mismatch")


class Evaluator:
    """对一条样本做执行等价判定。带 gold 结果缓存（同一 db+sql 只算一次）。"""

    def __init__(self, registry: DatabaseRegistry, *, timeout: float = 2.0,
                 sandbox_factory: type[SqlSandbox] = SqlSandbox):
        self.registry = registry
        self.timeout = timeout
        self._sandbox_factory = sandbox_factory
        self._gold_cache: dict[tuple[str, str], QueryResult] = {}

    def _run(self, db_id: str, sql: str) -> QueryResult:
        path = self.registry.db_path(db_id)
        sb = self._sandbox_factory(path, timeout=self.timeout)
        return sb.run(sql)

    def gold_result(self, db_id: str, gold_sql: str) -> QueryResult:
        key = (db_id, gold_sql)
        if key not in self._gold_cache:
            self._gold_cache[key] = self._run(db_id, gold_sql)
        return self._gold_cache[key]

    def evaluate(self, db_id: str, gold_sql: str, pred_sql: str) -> EvalResult:
        gold = self.gold_result(db_id, gold_sql)
        if not gold.success:
            return EvalResult(False, "gold_error", pred_rows=0)

        if not pred_sql or not pred_sql.strip():
            return EvalResult(False, "pred_error", gold_rows=gold.row_count,
                              pred_error_class="empty_answer")

        pred = self._run(db_id, pred_sql)
        if not pred.success:
            return EvalResult(False, "pred_error", gold_rows=gold.row_count,
                              pred_error_class=pred.error_class or "unknown")

        ok, reason = results_equal(gold, pred,
                                   gold_has_order=bool(_ORDER_BY_RE.search(gold_sql)))
        return EvalResult(ok, reason, gold_rows=gold.row_count, pred_rows=pred.row_count)
