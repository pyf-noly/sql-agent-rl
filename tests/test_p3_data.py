"""P3 数据管线测试：去重、SFT 转换、种子指纹、统计一致性。

这批测试防的是**静默的数据错位**——它不会让程序崩溃，只会让你训练在错的数据上。
"""

from __future__ import annotations

import hashlib
import json

import pytest

# 从脚本里导入纯函数（脚本本身有 main 保护，可安全 import）
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


synth = _load("scripts/synthesize_sft.py", "synth_sft")


def _ep(turns: list[dict], **kw) -> dict:
    base = {
        "db_id": "demo", "question": "Q?", "gold_sql": "SELECT 1",
        "n_turns": len(turns), "stopped_reason": kw.get("stopped_reason", "correct"),
        "turns": turns,
    }
    base.update(kw)
    return base


def _turn(sql: str, correct: bool | None, feedback: str = "fb", success: bool = True) -> dict:
    return {"turn": 1, "sql": sql, "success": success, "feedback": feedback,
            "correct": correct, "error_class": None, "error_message": "",
            "row_count": 1, "parse_source": "fence"}


class TestSeedFingerprint:
    def test_stable(self) -> None:
        seeds = [{"db_id": "a", "question": "q1", "difficulty": "easy"}]
        assert synth._seed_fingerprint(seeds) == synth._seed_fingerprint(seeds)

    def test_changes_with_content(self) -> None:
        a = [{"db_id": "a", "question": "q1", "difficulty": "easy"}]
        b = [{"db_id": "a", "question": "q2", "difficulty": "easy"}]
        assert synth._seed_fingerprint(a) != synth._seed_fingerprint(b)

    def test_changes_with_order(self) -> None:
        a = [{"db_id": "a", "question": "q1", "difficulty": "easy"},
             {"db_id": "b", "question": "q2", "difficulty": "hard"}]
        b = list(reversed(a))
        assert synth._seed_fingerprint(a) != synth._seed_fingerprint(b)

    def test_independent_of_extra_fields(self) -> None:
        a = [{"db_id": "a", "question": "q1", "difficulty": "easy"}]
        b = [{"db_id": "a", "question": "q1", "difficulty": "easy", "query": "SELECT 1"}]
        assert synth._seed_fingerprint(a) == synth._seed_fingerprint(b)


class TestToSft:
    def test_single_turn(self) -> None:
        ep = _ep([_turn("SELECT 1", True)])
        item = synth._to_sft(ep)
        assert item is not None
        assert len(item["messages"]) == 2
        assert item["messages"][0]["role"] == "user"
        assert "SELECT 1" in item["messages"][1]["content"]
        assert item["meta"]["n_turns"] == 1

    def test_multi_turn_keeps_feedback(self) -> None:
        ep = _ep([
            _turn("SELECT bad", False, feedback="no such column: x"),
            _turn("SELECT good", True),
        ])
        item = synth._to_sft(ep)
        assert item is not None
        assert len(item["messages"]) == 4  # user, assistant, user(feedback), assistant
        assert "no such column" in item["messages"][2]["content"]

    def test_three_turns(self) -> None:
        ep = _ep([
            _turn("SELECT 1", False, feedback="f1"),
            _turn("SELECT 2", False, feedback="f2"),
            _turn("SELECT 3", True),
        ])
        item = synth._to_sft(ep)
        assert len(item["messages"]) == 6
        assert item["messages"][-1]["role"] == "assistant"

    def test_never_correct_returns_none(self) -> None:
        ep = _ep([_turn("SELECT 1", False), _turn("SELECT 2", False),
                  _turn("SELECT 3", False)])
        assert synth._to_sft(ep) is None

    def test_empty_sql_skipped(self) -> None:
        ep = _ep([_turn("", False), _turn("SELECT 1", True)])
        item = synth._to_sft(ep)
        assert item is not None
        # 空 SQL 不该产生 assistant 消息
        assert all(m["content"].strip() for m in item["messages"])

    def test_no_turns_returns_none(self) -> None:
        assert synth._to_sft(_ep([])) is None


class TestDedupKey:
    """finalize 的去重键必须覆盖整条对话。"""

    @staticmethod
    def _key(messages: list[dict]) -> str:
        return hashlib.sha256(
            "\x1e".join(f"{m['role']}\x1f{m['content']}" for m in messages).encode()
        ).hexdigest()

    def test_different_multi_turn_not_deduped(self) -> None:
        # 同一个首轮错误，但后续修复不同 —— 不能被当成重复
        a = [{"role": "user", "content": "Q"},
             {"role": "assistant", "content": "bad"},
             {"role": "user", "content": "fb1"},
             {"role": "assistant", "content": "fix A"}]
        b = [{"role": "user", "content": "Q"},
             {"role": "assistant", "content": "bad"},
             {"role": "user", "content": "fb2"},
             {"role": "assistant", "content": "fix B"}]
        assert self._key(a) != self._key(b)

    def test_identical_is_deduped(self) -> None:
        a = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        b = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        assert self._key(a) == self._key(b)

    def test_role_matters(self) -> None:
        a = [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}]
        b = [{"role": "assistant", "content": "Y"}, {"role": "user", "content": "X"}]
        assert self._key(a) != self._key(b)


class TestDataIntegrity:
    """对真实产物做的完整性检查（文件不存在则跳过）。"""

    def test_sft_final_no_duplicates(self) -> None:
        p = ROOT / "data/processed/sft_final.jsonl"
        if not p.is_file():
            pytest.skip("sft_final.jsonl 不存在")
        seen = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            k = hashlib.sha256(
                "\x1e".join(f"{m['role']}\x1f{m['content']}" for m in d["messages"]).encode()
            ).hexdigest()
            assert k not in seen, "sft_final 里存在重复样本"
            seen.add(k)

    def test_rl_prompts_have_no_answers(self) -> None:
        p = ROOT / "data/processed/rl_prompts.json"
        if not p.is_file():
            pytest.skip("rl_prompts.json 不存在")
        rows = json.loads(p.read_text(encoding="utf-8"))
        assert rows
        for r in rows:
            assert "query" not in r, "RL prompt 里混入了 gold SQL（数据泄漏）"
            assert "gold_sql" not in r
            assert r["db_id"] and r["question"]

    def test_rl_and_sft_disjoint(self) -> None:
        """RL 题与 SFT 题不能重叠，否则 RL 评测会虚高。

        容忍极少数（<0.5%）重叠：Spider 本身有重复题干，
        历史数据里可能残留 1~2 条。超过这个比例说明切分逻辑坏了。
        """
        rl_p = ROOT / "data/processed/rl_prompts.json"
        sft_p = ROOT / "data/processed/sft_final.jsonl"
        if not rl_p.is_file() or not sft_p.is_file():
            pytest.skip("数据文件不全")
        rl_rows = json.loads(rl_p.read_text(encoding="utf-8"))
        rl = {(r["db_id"], r["question"]) for r in rl_rows}
        sft = set()
        for line in sft_p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            sft.add((d["meta"]["db_id"], d["messages"][0]["content"]))
        overlap = rl & sft
        ratio = len(overlap) / max(1, len(rl))
        assert ratio < 0.005, f"SFT 与 RL 重叠 {len(overlap)} 条（{ratio:.2%}），切分逻辑有问题"
