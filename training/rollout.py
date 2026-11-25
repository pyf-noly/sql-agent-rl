"""Rollout：让本地模型替掉 API 客户端，把 ``SqlAgent`` 原样跑起来。

关键设计：**训练时的交互与评测时完全一致**。
``SqlAgent`` 是 P1 写的，P2 的三套基线、P3 的轨迹合成都走它；P4 训练如果另写一套
agent 循环，"提升"就可能来自循环差异而不是算法。所以这里只做一件事：
实现一个长得跟 ``LLMClient`` 一样的 ``.chat()``，其余全部复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from envs.db_registry import DatabaseRegistry
from rollout.agent import Episode, SqlAgent
from rollout.evaluator import Evaluator

# 注意：这里**不要**在模块顶层 import training.model——它一上来就 import torch，
# 会让"只测奖励函数"的单测也得装 torch。生成那段改成在 chat() 里惰性导入。


@dataclass
class LocalUsage:
    """与 ``rollout.llm_client.Usage`` 同构，但本地生成没有钱的花费。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def cost_cny(self) -> float:
        return 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "cost_cny": 0.0,
        }


@dataclass
class HFPolicyClient:
    """把本地 HF 模型包装成 OpenAI 客户端的样子（只需要 ``.chat``）。"""

    model: object
    tokenizer: object
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    name: str = "local"
    base_url: str = "local"
    usage: LocalUsage = field(default_factory=LocalUsage)

    @property
    def model_name(self) -> str:
        """日志与结果文件里用的模型标识。

        绝不能把 ``self.model``（一个 PeftModel）直接塞进 JSON——那会在写结果
        的最后一行抛 TypeError，把几十分钟的评测全部作废。
        """
        return self.name

    def chat(self, messages: list[dict], **_: object) -> str:
        from training.model import generate_text  # 惰性导入，避免顶层依赖 torch

        self.model.eval()  # 采样时必须是 eval（否则 dropout 会让 logp 对不上）
        text = generate_text(
            self.model, self.tokenizer, messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self.usage.calls += 1
        self.usage.completion_tokens += max(1, len(text) // 4)
        self.usage.prompt_tokens += sum(len(m.get("content", "")) for m in messages) // 4
        return text


def make_agent(client, registry: DatabaseRegistry, *, max_turns: int,
               timeout: float = 2.0, temperature: float = 0.9,
               prompt_cfg=None, max_rows: int = 1000,
               compare: str = "rows", value_check: bool = False) -> tuple[SqlAgent, Evaluator]:
    """造一个"判对才停"的 agent（也就是 B2 的交互方式，训练用同一套）。

    ``prompt_cfg`` 与评测端必须一致（BIRD 传 ``BIRD_PROMPT``）；``max_rows`` /
    ``compare`` / ``value_check`` 同理——反馈与判定口径一旦两端不一致，
    训练出来的行为在评测时就对不上。
    """
    ev = Evaluator(registry, timeout=timeout, max_rows=max_rows, compare=compare)
    agent = SqlAgent(
        client, registry,
        max_turns=max_turns,
        timeout=timeout,
        temperature=temperature,
        stop_on_success=False,      # 训练时必须跑满轮数，否则学不到修复
        stop_on_correct=True,       # 判对就停，省生成
        verifier=ev,
        prompt_cfg=prompt_cfg,
        max_rows=max_rows,
        value_check=value_check,
    )
    return agent, ev


def episode_reward(ep: Episode, sample: dict, ev: Evaluator, *,
                   correct: float = 1.0, wrong: float = 0.0,
                   parse_error: float = -0.2,
                   turn_penalty: float = 0.0,
                   first_turn_bonus: float = 0.0) -> tuple[float, dict]:
    """一条轨迹的奖励。

    默认是 **outcome-only**（只看最终 SQL 对不对）。P5 实测发现：当模型已达 86%，
    G=4 的组里 90% 是"全对或全错"，组内优势恒为 0，梯度为零。

    所以这里额外提供两个**轮级塑形项**，让"全对组"也有方差：

    - ``turn_penalty``：每多用一轮扣多少分。全对组里"1 轮解出"得 1.0、
      "3 轮才解出"得 1.0-2λ，组内立刻有高低之分；
    - ``first_turn_bonus``：一轮就对的额外奖励（更激进的做法）。

    这两项只在"答对"时生效——答错的轮数没有意义，不该被奖励。
    注意它们是**奖励塑形**，不是过程奖励：P6 才会把信号拆到每一轮。
    """
    info: dict = {
        "n_turns": ep.n_turns,
        "stopped_reason": ep.stopped_reason,
        "pred_error_class": None,
    }
    if not ep.final_sql:
        r = parse_error if ep.stopped_reason == "parse_error" else wrong
        info["correct"] = False
        info["reason"] = ep.stopped_reason or "no_sql"
        return r, info

    res = ev.evaluate(sample["db_id"], sample["query"], ep.final_sql)
    info.update(correct=bool(res.correct), reason=res.reason,
                pred_error_class=res.pred_error_class)
    if not res.correct:
        return wrong, info

    r = correct
    if turn_penalty:
        r -= turn_penalty * max(0, ep.n_turns - 1)
    if first_turn_bonus and ep.n_turns == 1:
        r += first_turn_bonus
    info["reward_shaping"] = round(r - correct, 4)
    return r, info
