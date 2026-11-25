"""把「对话」变成「token 序列 + loss mask」——训练与推理共用同一套模板。

这个模块只有一个职责，但它是整个 RL 训练最容易出错的地方，所以单独拿出来讲清楚。

为什么需要 loss mask
-------------------
一条多轮轨迹长这样::

    [system][user: schema+问题] [assistant: SQL] [user: 执行反馈] [assistant: SQL]

只有 ``assistant`` 的 token 是**模型自己生成的**，也只有它们该参与 loss/策略梯度。
把 user 的 schema、工具返回的反馈也算进去，等于让模型去"预测别人说的话"，
轻则不收敛，重则模型学会复读反馈里的表名。

为什么不能自己拼字符串
---------------------
Chat 模板不只是 ``"<|im_start|>" + role``，它还会加 BOS、空格、EOS，
不同模型还不一样（Qwen3 的模板还带 ``enable_thinking`` 开关）。
所以这里一律走 ``tokenizer.apply_chat_template``，绝不手写拼接。

mask 的边界怎么定（关键算法）
-----------------------------
对第 i 条 assistant 消息::

    start = len(apply_chat_template(messages[:i], add_generation_prompt=True))
    end   = len(apply_chat_template(messages[:i+1], add_generation_prompt=False))

``[start, end)`` 就是"这一轮助手说的话 + 结束符"。用两个模板长度相减来定边界，
比"找 ```sql 出现的位置"稳得多——模板改了、模型输出带了解释性文字，都不会错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 模型输出里必须出现的代码块标记（P1 的解析器依赖它）
SQL_FENCE = "```sql"


def template_kwargs(tokenizer: Any) -> dict[str, Any]:
    """按模板能力决定要不要传 ``enable_thinking``。

    Qwen3 系列的模板支持 ``enable_thinking``：关掉它，模型直接输出 SQL，
    不浪费几百个 token 去"思考"。但 Instruct-2507 这类非思考模型、以及老模板
    不认识这个参数，传了会抛异常，所以这里先看模板源码再决定。
    """
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(tmpl, dict):  # 某些 tokenizer 存成 {name: template}
        tmpl = " ".join(str(v) for v in tmpl.values())
    if "enable_thinking" in tmpl:
        return {"enable_thinking": False}
    return {}


class _TemplateCache:
    """给 tokenizer 缓存一份 kwargs，避免每条样本都去读模板字符串。"""

    def __init__(self) -> None:
        self._by_id: dict[int, dict[str, Any]] = {}

    def get(self, tokenizer: Any) -> dict[str, Any]:
        key = id(tokenizer)
        if key not in self._by_id:
            self._by_id[key] = template_kwargs(tokenizer)
        return self._by_id[key]


_TEMPLATE_CACHE = _TemplateCache()


def apply_template(
    tokenizer: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    extra: dict[str, Any] | None = None,
) -> list[int]:
    """调 ``apply_chat_template`` 并统一返回 ``list[int]``。

    带两层兼容：

    1. 模板不认 ``enable_thinking`` 时去掉重试一次，而不是让训练崩掉；
    2. **transformers 5.x 在 ``tokenize=True, return_tensors=None`` 时返回
       ``BatchEncoding``（含 input_ids / attention_mask），4.x 返回 list[int]**。
       直接 ``list(out)`` 在 5.x 上会变成 ``['input_ids', 'attention_mask']``，
       再塞进 ``torch.tensor`` 就是 ``ValueError: too many dimensions 'str'``
       （2026-11 在本地 Windows + transformers 5.22 上踩到，租卡机是 4.x 所以不报）。
    """
    kwargs = {**_TEMPLATE_CACHE.get(tokenizer), **(extra or {})}
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            return_tensors=None, **kwargs)
    except (TypeError, ValueError) as e:  # 模板不支持某个参数
        if not kwargs:
            raise
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            return_tensors=None)
    return _as_id_list(out)


def _as_id_list(out: Any) -> list[int]:
    """把 4.x 的 list[int] / 5.x 的 BatchEncoding / dict 统一成 list[int]。"""
    if hasattr(out, "input_ids"):          # transformers >= 5 的 BatchEncoding
        out = out["input_ids"]
    elif isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):  # 意外多了一层 batch 维
        out = out[0]
    return [int(x) for x in out]


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    """纯文本编码，不加任何特殊符号（因为模板已经加过了）。"""
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    return list(ids)


@dataclass
class Encoded:
    """一条训练样本的编码结果。"""

    input_ids: list[int]
    loss_mask: list[int]
    """1 = 参与 loss（助手生成）；0 = 上下文（system/user/工具反馈）。"""

    turn_spans: list[tuple[int, int]] = field(default_factory=list)
    """每个 assistant 轮次在序列里的 ``[start, end)`` 区间（按轮次顺序）。

    P6 的轮级信用分配要用它把"这一轮的优势"铺到对应的 token 上；
    没有它就只能整条轨迹一个优势值，轮级信号无从谈起。
    """

    n_loss_tokens: int = 0
    prompt_len: int = 0
    """第一个助手 token 之前的前缀长度，用于统计与截断诊断。"""

    n_mismatch: int = 0
    """mask 区间与助手原文 token 对不上的次数。正常应恒为 0，>0 说明模板有蹊跷。"""

    n_truncated: int = 0
    """因超长被丢弃的 token 数。丢掉的如果含助手 token，loss 会变少。"""

    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def context_ratio(self) -> float:
        """上下文占比。这个数越大多轮训练越"贵"，P4 验收要看它。"""
        return 1.0 - (self.n_loss_tokens / self.length) if self.length else 0.0

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "prompt_len": self.prompt_len,
            "n_loss_tokens": self.n_loss_tokens,
            "context_ratio": round(self.context_ratio, 4),
            "n_mismatch": self.n_mismatch,
            "n_truncated": self.n_truncated,
        }


def encode_with_spans(tokenizer: Any, messages: list[dict], *,
                      max_len: int | None = None) -> Encoded:
    """把 messages 编码成 ``input_ids + loss_mask``（见模块头部的算法说明）。"""
    full = apply_template(tokenizer, messages, add_generation_prompt=False)
    mask = [0] * len(full)

    n_mismatch = 0
    prompt_len = 0
    seen_assistant = False
    spans: list[tuple[int, int]] = []

    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        start = len(apply_template(tokenizer, messages[:i], add_generation_prompt=True))
        end = len(apply_template(tokenizer, messages[: i + 1], add_generation_prompt=False))
        if not seen_assistant:
            prompt_len = start
            seen_assistant = True
        if not (0 <= start < end <= len(full)):
            n_mismatch += 1
            continue
        mask[start:end] = [1] * (end - start)
        spans.append((start, end))
        # 自检：这段区间的开头必须就是助手原文（EOS 之类的尾巴不算）
        text_ids = tokenize_text(tokenizer, m.get("content", ""))
        if text_ids and full[start:start + len(text_ids)] != text_ids:
            n_mismatch += 1

    n_truncated = 0
    if max_len and len(full) > max_len:
        n_truncated = len(full) - max_len
        full = full[:max_len]
        mask = mask[:max_len]
        spans = [(min(s, max_len), min(e, max_len)) for s, e in spans if s < max_len]

    enc = Encoded(
        input_ids=full,
        loss_mask=mask,
        turn_spans=spans,
        n_loss_tokens=sum(mask),
        prompt_len=prompt_len,
        n_mismatch=n_mismatch,
        n_truncated=n_truncated,
    )
    enc.meta["n_messages"] = len(messages)
    enc.meta["n_assistant"] = sum(1 for m in messages if m.get("role") == "assistant")
    return enc


def messages_from_episode(episode: Any, schema: Any, prompt_cfg: Any = None) -> list[dict]:
    """把 ``Episode`` 还原成**推理时逐字一致**的 messages。

    依赖 ``Turn.feedback_message``（agent 保存的原始消息），不再自己拼模板——
    这样就消除了"训练一套模板、推理另一套模板"的隐患。

    ``prompt_cfg`` 与 ``episode.evidence`` 必须和 rollout 时一致：BIRD 的 prompt
    带 evidence/类型/引号，少传一个，训练序列就和采样时的分布不同了（静默变差）。
    """
    from rollout.prompt import build_messages

    messages = build_messages(episode.question, schema, prompt_cfg,
                              evidence=getattr(episode, "evidence", None))
    for t in episode.turns:
        messages.append({"role": "assistant", "content": t.raw_output})
        if t.feedback_message is not None:
            messages.append(t.feedback_message)
    return messages
