"""跑一条样本的完整交互：模型出 SQL → 沙箱执行 → 反馈回灌 → 重试（或停止）。

单轮和多轮共用这一个函数：``max_turns=1`` 就是基线 B1，``max_turns=3`` 就是 B2。
这样保证"多轮 vs 单轮"的差别**只来自轮数**，不来自两套不同的代码路径——
这是做消融的基本纪律。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from envs.db_registry import DatabaseRegistry, DbSchema
from envs.error_taxonomy import classify_result
from envs.feedback import FeedbackFormatter
from envs.sql_sandbox import SqlSandbox
from envs.value_check import check_literals, format_hints
from rollout.llm_client import LLMClient
from rollout.parse_sql import parse_sql
from rollout.prompt import PromptConfig, build_feedback_message, build_messages

if TYPE_CHECKING:
    from rollout.evaluator import Evaluator


@dataclass
class Turn:
    turn: int
    sql: str
    success: bool
    error_class: str | None = None
    error_message: str = ""
    row_count: int = 0
    raw_output: str = ""
    parse_source: str = ""
    feedback: str = ""
    feedback_message: dict[str, Any] | None = None
    """下一轮真正发给模型的那条 user 消息（原样保存）。

    训练时必须用它重建序列：如果训练时自己再拼一遍反馈模板，一旦模板改了，
    训练分布就和推理分布悄悄分叉——这类 bug 不会报错，只会让效果莫名其妙变差。
    """
    correct: bool | None = None
    """仅当 ``stop_on_correct=True`` 时填写：这一轮的答案是否符合 gold。"""


@dataclass
class Episode:
    db_id: str
    question: str
    gold_sql: str
    turns: list[Turn] = field(default_factory=list)
    final_sql: str = ""
    stopped_reason: str = ""   # success / max_turns / exec_error / parse_error / api_error
    api_error: str = ""
    evidence: str = ""
    """BIRD 的外部知识提示。必须随 Episode 带走：训练时要靠它重建**同一条** prompt，
    否则 rollout 用了 evidence、训练序列却没用 → 训练/推理静默分叉。"""

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def executed_ok(self) -> bool:
        """最后一轮是否执行成功（不代表答案对，只代表 SQL 能跑）。"""
        return bool(self.turns) and self.turns[-1].success

    def to_dict(self, *, truncate_feedback: bool = True) -> dict:
        """序列化。

        ``truncate_feedback=True`` 用于日志与人工检查（省体积）；
        合成 SFT 轨迹时必须传 False——否则模型学到的反馈是截断的。
        """
        return {
            "db_id": self.db_id,
            "question": self.question,
            "gold_sql": self.gold_sql,
            "final_sql": self.final_sql,
            "stopped_reason": self.stopped_reason,
            "api_error": self.api_error,
            "n_turns": self.n_turns,
            "turns": [
                {
                    "turn": t.turn,
                    "sql": t.sql,
                    "success": t.success,
                    "error_class": t.error_class,
                    "error_message": t.error_message[:500],
                    "row_count": t.row_count,
                    "parse_source": t.parse_source,
                    "feedback": (t.feedback[:500] if truncate_feedback else t.feedback),
                    "correct": t.correct,
                }
                for t in self.turns
            ],
        }


class SqlAgent:
    """把 LLM + 沙箱 + 反馈串成一个 agent。

    Parameters
    ----------
    max_turns:
        1 = 单轮基线；3 = 三轮自修复基线。
    stop_on_success:
        执行成功就停。注意：**执行成功 ≠ 答案正确**，所以训练时不能拿它当奖励，
        这里只是为了省 token。
    temperature:
        基线用 0（可复现）；训练采样用 >0。
    """

    def __init__(
        self,
        client: LLMClient,
        registry: DatabaseRegistry,
        *,
        max_turns: int = 1,
        timeout: float = 2.0,
        temperature: float = 0.0,
        stop_on_success: bool = True,
        formatter: FeedbackFormatter | None = None,
        verifier: "Evaluator | None" = None,
        stop_on_correct: bool = False,
        prompt_cfg: PromptConfig | None = None,
        max_rows: int = 1000,
        value_check: bool = False,
    ):
        """Parameters
        ----------
        stop_on_success:
            SQL 能执行就停。**默认 True，但这对多轮没意义**——Spider 的 SQL 几乎
            都能执行，模型永远不会进第 2 轮。要测多轮上限必须配 ``stop_on_correct``。
        verifier:
            用来判"答案对不对"的评测器。只有 ``stop_on_correct=True`` 时才需要。
            注意：它**只用于决定是否继续**，绝不能把正确性信息写进反馈——
            否则就是给模型开卷考试，测出来的不是多轮能力，是作弊能力。
        stop_on_correct:
            True 时"答案判对才停"，用于测多轮自修复的**上限**（B2）。
        prompt_cfg:
            提示词配置。默认 ``None`` 等价于 Spider 的旧行为（不带类型/外键）；
            BIRD 传 :data:`rollout.prompt.BIRD_PROMPT`。
        max_rows:
            执行结果的最大行数（超过即截断）。BIRD 有 55 条 gold 会查出 >1000 行，
            评测时要放大，否则"截断后的前缀相同"会掩盖差异。
        value_check:
            是否在"执行成功"时额外检查 WHERE 里的字符串字面量是否真的存在于库中。
            这是唯一能**不泄漏答案**地把"语义错误"变成可检测信号的手段（见
            ``envs/value_check.py``）。默认 False，保证 P4/P5 的历史结果可复现。
        """
        self.client = client
        self.registry = registry
        self.max_turns = max_turns
        self.timeout = timeout
        self.stop_on_success = stop_on_success
        self.formatter = formatter or FeedbackFormatter()
        self.verifier = verifier
        self.stop_on_correct = stop_on_correct
        self.prompt_cfg = prompt_cfg
        self.max_rows = max_rows
        self.value_check = value_check
        if stop_on_correct and verifier is None:
            raise ValueError("stop_on_correct=True 需要传入 verifier")

    def _schema(self, db_id: str) -> DbSchema:
        return self.registry.schema(db_id)

    def run(self, sample: dict[str, Any]) -> Episode:
        db_id = sample["db_id"]
        question = sample["question"]
        gold = sample.get("query", "")

        ep = Episode(db_id=db_id, question=question, gold_sql=gold,
                     evidence=sample.get("evidence") or "")

        schema = self._schema(db_id)
        # BIRD 的样本会带 evidence（外部知识提示）；Spider 没有这个字段，
        # 传 None 时 prompt 与旧版逐字一致。
        messages = build_messages(question, schema, self.prompt_cfg,
                                  evidence=sample.get("evidence"))
        sandbox = SqlSandbox(self.registry.db_path(db_id), timeout=self.timeout,
                             max_rows=self.max_rows)

        for t in range(1, self.max_turns + 1):
            try:
                raw = self.client.chat(messages)
            except Exception as e:  # noqa: BLE001
                ep.api_error = f"{type(e).__name__}: {e}"
                ep.stopped_reason = "api_error"
                break

            parsed = parse_sql(raw)
            if not parsed.ok:
                fb_msg = build_feedback_message(
                    "(no SQL found)",
                    "Your reply did not contain a ```sql code block with a SELECT "
                    "statement. Reply with only the SQL inside a ```sql block.",
                )
                turn = Turn(turn=t, sql="", success=False, error_class="parse_error",
                            raw_output=raw, parse_source=parsed.source,
                            feedback_message=fb_msg)
                ep.turns.append(turn)
                ep.final_sql = ""
                ep.stopped_reason = "parse_error"
                # 给模型一次"没抽出 SQL"的反馈，再试
                messages = messages + [{"role": "assistant", "content": raw}, fb_msg]
                continue

            result = sandbox.run(parsed.sql)
            cls = classify_result(result.success, result.error_class,
                                  result.error_message, result.is_empty)

            if result.success:
                feedback = self.formatter.render_success(result)
                if self.value_check:
                    # 只在"执行成功"时补值检查：执行报错本身信息量已经够了。
                    # 它不改变判定，只给模型"这个取值在库里不存在"的依据。
                    try:
                        hints = format_hints(check_literals(parsed.sql, schema,
                                                            timeout=self.timeout))
                        if hints:
                            feedback = feedback + "\n" + hints
                    except Exception as e:  # noqa: BLE001  值检查失败不该影响主流程
                        feedback = feedback + f"\n(value check skipped: {type(e).__name__})"
            else:
                feedback = self.formatter.render(
                    result, cls,
                    table_names=schema.table_names,
                    columns=schema.columns_by_table,
                )

            # 判"答案对不对"。只用于决定要不要继续，绝不回灌给模型。
            is_correct: bool | None = None
            if self.stop_on_correct and result.success and self.verifier is not None:
                is_correct = self.verifier.evaluate(db_id, gold, parsed.sql).correct

            turn = Turn(
                turn=t, sql=parsed.sql, success=result.success,
                error_class=(result.error_class or cls.category.value) if not result.success else None,
                error_message=result.error_message,
                row_count=result.row_count,
                raw_output=raw, parse_source=parsed.source,
                feedback=feedback,
                correct=is_correct,
            )
            ep.turns.append(turn)
            ep.final_sql = parsed.sql

            if self.stop_on_correct:
                # 多轮上限模式：判对才停；判错则继续，但反馈里只有执行结果
                if is_correct:
                    ep.stopped_reason = "correct"
                    break
                if t == self.max_turns:
                    ep.stopped_reason = "max_turns"
                    break
                fb_msg = build_feedback_message(parsed.sql, feedback)
                turn.feedback_message = fb_msg
                messages = messages + [{"role": "assistant", "content": raw}, fb_msg]
                continue

            if result.success and self.stop_on_success:
                ep.stopped_reason = "success"
                break
            if t == self.max_turns:
                ep.stopped_reason = "max_turns" if not result.success else "success"
                break

            fb_msg = build_feedback_message(parsed.sql, feedback)
            turn.feedback_message = fb_msg
            messages = messages + [{"role": "assistant", "content": raw}, fb_msg]

        return ep
