"""BIRD 数据管线的单测：字段映射、DDL 引号、evidence 进 prompt、max_rows。

这四件事都是"错了不会报错、只会让分数莫名其妙变低"的类型：
字段名映射错 → gold 为空 → 全判错；DDL 不给带空格的列名加引号 → 模型自己删空格
→ SQL 语法错；evidence 没进 prompt → BIRD 的外部知识白给；max_rows 太小 →
两边都截断到同样前缀 → **假阳性**（错 SQL 判对）。
"""

from __future__ import annotations

import json

import pytest

from envs.db_registry import DatabaseRegistry, DbSchema, TableInfo, load_bird_examples
from envs.sql_sandbox import SqlSandbox
from rollout.agent import SqlAgent
from rollout.evaluator import Evaluator
from rollout.prompt import BIRD_PROMPT, PromptConfig, build_messages


class TestLoadBird:
    def test_maps_fields(self, tmp_path) -> None:
        p = tmp_path / "dev.json"
        p.write_text(json.dumps([
            {"question_id": 1, "db_id": "financial", "question": "q1",
             "evidence": " T2 is loans ", "SQL": "SELECT 1", "difficulty": "simple"},
            {"question_id": 2, "db_id": "financial", "question": "q2",
             "evidence": "", "SQL": "SELECT 2", "difficulty": "challenging"},
        ]), encoding="utf-8")
        rows = load_bird_examples(p)
        assert rows[0]["query"] == "SELECT 1"          # SQL → query
        assert rows[0]["evidence"] == "T2 is loans"    # 去掉首尾空白
        assert rows[0]["difficulty"] == "easy"         # simple → easy
        assert rows[1]["difficulty"] == "hard"
        assert rows[1]["difficulty_raw"] == "challenging"

    def test_missing_sql_raises(self, tmp_path) -> None:
        p = tmp_path / "dev.json"
        p.write_text(json.dumps([{"question_id": 1, "db_id": "x", "question": "q"}]),
                     encoding="utf-8")
        with pytest.raises(KeyError):
            load_bird_examples(p)


class TestDdlQuoting:
    def test_quotes_identifiers_with_spaces(self) -> None:
        schema = DbSchema(db_id="d", path=None, tables=(
            TableInfo("frpm", ("CDSCode", "County Name", "Free Meal Count (K-12)"),
                      types=("TEXT", "TEXT", "REAL")),
        ))
        ddl = schema.render_ddl(10, include_types=True, quote_identifiers=True)
        assert '"County Name" TEXT' in ddl
        assert '"Free Meal Count (K-12)" REAL' in ddl
        assert "CDSCode TEXT" in ddl

    def test_no_quotes_by_default(self) -> None:
        schema = DbSchema(db_id="d", path=None, tables=(TableInfo("t", ("a b",)),))
        assert '"a b"' not in schema.render_ddl()

    def test_column_budget_still_respected(self) -> None:
        cols = tuple(f"c{i}" for i in range(30))
        schema = DbSchema(db_id="d", path=None, tables=(TableInfo("t", cols),))
        ddl = schema.render_ddl(5)
        assert ddl.count("c") >= 1 and ", ..." in ddl

    def test_types_and_fk(self) -> None:
        schema = DbSchema(db_id="d", path=None, tables=(
            TableInfo("a", ("id",), types=("INTEGER",), foreign_keys=(("x", "b", "y"),)),
        ))
        ddl = schema.render_ddl(10, include_types=True, include_fk=True)
        assert "id INTEGER" in ddl
        assert "-- FK: a.x -> b.y" in ddl


class TestEvidenceInPrompt:
    def test_evidence_section_present(self, tmp_path) -> None:
        schema = DbSchema(db_id="d", path=None, tables=(TableInfo("t", ("a",)),))
        msgs = build_messages("q", schema, BIRD_PROMPT, evidence="T2 means rating")
        assert "### External knowledge" in msgs[1]["content"]
        assert "T2 means rating" in msgs[1]["content"]

    def test_no_section_without_evidence(self) -> None:
        schema = DbSchema(db_id="d", path=None, tables=(TableInfo("t", ("a",)),))
        assert "External knowledge" not in build_messages("q", schema)[1]["content"]

    def test_evidence_can_be_disabled(self) -> None:
        schema = DbSchema(db_id="d", path=None, tables=(TableInfo("t", ("a",)),))
        cfg = PromptConfig(include_evidence=False)
        msgs = build_messages("q", schema, cfg, evidence="secret hint")
        assert "secret hint" not in msgs[1]["content"]


class _EchoClient:
    """假的 LLM：把收到的 user prompt 存下来，回一个固定 SQL。"""

    model_name = "echo"
    base_url = "test"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, messages, **_) -> str:
        self.prompts.append(messages[1]["content"])
        return "```sql\nSELECT 1\n```"


class TestAgentEvidence:
    def test_agent_passes_evidence_into_prompt(self, db_root) -> None:
        reg = DatabaseRegistry(db_root)
        client = _EchoClient()
        agent = SqlAgent(client, reg, max_turns=1, prompt_cfg=BIRD_PROMPT)
        sample = {"db_id": "company_1", "question": "how many?",
                  "query": "SELECT 1", "evidence": "dept_id 1 = Engineering"}
        agent.run(sample)
        assert "dept_id 1 = Engineering" in client.prompts[0]
        assert "### External knowledge" in client.prompts[0]

    def test_episode_keeps_evidence_for_training(self, db_root) -> None:
        """训练序列靠 Episode.evidence 重建 prompt；丢了就等于训练/采样分叉。"""
        reg = DatabaseRegistry(db_root)
        sample = {"db_id": "company_1", "question": "q", "query": "SELECT 1",
                  "evidence": "hint-xyz"}
        ep = SqlAgent(_EchoClient(), reg, max_turns=1, prompt_cfg=BIRD_PROMPT).run(sample)
        assert ep.evidence == "hint-xyz"
        from training.chatfmt import messages_from_episode

        msgs = messages_from_episode(ep, reg.schema("company_1"), BIRD_PROMPT)
        assert "hint-xyz" in msgs[1]["content"]
        # 不传 prompt_cfg 时：evidence 仍会带（default 就有），但 DDL 回到 Spider 形态
        # （无类型、无引号）——这正是"少传一个参数就静默分叉"的地方。
        msgs2 = messages_from_episode(ep, reg.schema("company_1"))
        assert "hint-xyz" in msgs2[1]["content"]
        assert "dept_id INTEGER" not in msgs2[1]["content"]


class TestCompareModes:
    """两种等价口径必须各自可复现：rows（Spider 历史）与 bird（官方 set）。"""

    def test_rows_mode_is_row_count_sensitive(self) -> None:
        from envs.sql_sandbox import QueryResult
        from rollout.evaluator import results_equal

        g = QueryResult(True, ["a"], [(1,), (1,)])
        p = QueryResult(True, ["a"], [(1,)])
        assert results_equal(g, p, gold_has_order=False)[0] is False
        assert results_equal(g, p, gold_has_order=False, mode="rows")[0] is False

    def test_bird_mode_ignores_duplicates_and_order(self) -> None:
        from envs.sql_sandbox import QueryResult
        from rollout.evaluator import results_equal

        g = QueryResult(True, ["a"], [(1,), (2,)])
        p = QueryResult(True, ["a"], [(2,), (1,), (1,)])
        assert results_equal(g, p, gold_has_order=False, mode="bird")[0] is True

    def test_bird_mode_is_column_order_sensitive(self) -> None:
        from envs.sql_sandbox import QueryResult
        from rollout.evaluator import results_equal

        g = QueryResult(True, ["a", "b"], [(1, "x")])
        p = QueryResult(True, ["b", "a"], [("x", 1)])
        assert results_equal(g, p, gold_has_order=False, mode="bird")[0] is False
        # 同一组数据在 rows 口径下被认为是等价的（列顺序被忽略）
        assert results_equal(g, p, gold_has_order=False, mode="rows")[0] is True

    def test_bad_compare_raises(self, db_root) -> None:
        reg = DatabaseRegistry(db_root)
        with pytest.raises(ValueError):
            Evaluator(reg, compare="nope")


class TestMaxRows:
    def test_large_max_rows_not_truncated(self, tiny_db) -> None:
        sql = "SELECT e.emp_name FROM employee e JOIN employee f ON 1=1"
        small = SqlSandbox(tiny_db, max_rows=4).run(sql)
        large = SqlSandbox(tiny_db, max_rows=10000).run(sql)
        assert small.truncated is True
        assert large.truncated is False
        assert large.row_count > small.row_count

    def test_evaluator_accepts_max_rows(self, db_root) -> None:
        reg = DatabaseRegistry(db_root)
        ev = Evaluator(reg, max_rows=20000)
        r = ev.evaluate("company_1", "SELECT emp_name FROM employee", "SELECT emp_name FROM employee")
        assert r.correct
