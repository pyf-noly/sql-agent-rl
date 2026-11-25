"""Prompt 拼装：把 schema + 问题拼成模型输入，并约束输出格式。

为什么要单独一个模块
--------------------
prompt 是这个项目的**实验变量之一**。P2 的基线、P3 的合成、P5 的训练，
用的必须是同一个 prompt；否则"提升"可能只是 prompt 变好了。

所以这里的原则是：
1. 只有一套模板，所有阶段复用；
2. 模板可参数化（要不要 few-shot、要不要加列类型、要不要加示例值），
   但默认配置固定，实验之间不许偷偷改；
3. 输出格式约束写死：模型必须回 ```sql 代码块——P1 的解析器依赖这一点。

BIRD 的差异（2026-11 加入）
--------------------------
Spider 的 prompt 只有 schema + 问题，这套默认值**一个字都不动**（P2~P5 的数字
才能继续比较）。BIRD 需要两样额外信息，都通过 :class:`PromptConfig` 显式开启：

- ``evidence``：官方给的外部知识提示（"T2 表示用户评分"这类），是 BIRD 的
  标准设定，不是答案泄漏；
- 列类型 + 外键：BIRD 的库列多、列名多是缩写，不给类型基本没法猜。
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.db_registry import DbSchema

#: 系统提示：定死"你是 SQL 专家 + 只输出 sql 块"
SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a question, "
    "write ONE SQLite query that answers the question.\n"
    "Rules:\n"
    "- Reply with exactly one `SELECT` (or `WITH ... SELECT`) statement.\n"
    "- Put the SQL inside a ```sql code block. No other text.\n"
    "- Use only the tables and columns shown in the schema.\n"
    "- Do not invent columns; check the spelling carefully.\n"
)


@dataclass(slots=True)
class PromptConfig:
    """prompt 的可调项。默认值就是 P2~P6 全程使用的配置。"""

    include_ddl: bool = True
    include_column_types: bool = False   # Spider 的 schema 里类型信息不可靠，默认关
    include_fk: bool = False             # 外键注释（BIRD 打开，Spider 默认关）
    include_evidence: bool = True        # 有 evidence 才拼；Spider 没有这个字段，等价于不变
    quote_identifiers: bool = False      # 列名含空格/括号时加双引号（BIRD 必须开）
    max_schema_chars: int = 6000         # 防超长 schema 撑爆上下文
    max_schema_cols: int = 60            # 列预算；BIRD 的库最多 199 列，要放大到 400
    few_shot: int = 0                    # 默认零样本


#: BIRD 用的 preset：显式打开类型/外键/evidence/引号，并把列预算放大。
#: 单独一个常量而不是散在调用处，是为了让"BIRD 的 prompt 长什么样"可被 grep 到。
BIRD_PROMPT = PromptConfig(
    include_column_types=True,
    include_fk=True,
    include_evidence=True,
    quote_identifiers=True,
    max_schema_cols=400,
)


def build_user_prompt(question: str, schema: DbSchema, cfg: PromptConfig | None = None,
                      *, evidence: str | None = None) -> str:
    """拼用户消息：schema +（可选）外部知识 + 问题 + 输出要求。

    ``evidence`` 为 BIRD 的外部知识提示；为空时与 Spider 的旧 prompt 完全一致
    （同样两段、同样的措辞），保证历史结果可复现。
    """
    cfg = cfg or PromptConfig()
    parts: list[str] = []

    if cfg.include_ddl:
        ddl = schema.render_ddl(cfg.max_schema_cols,
                                include_types=cfg.include_column_types,
                                include_fk=cfg.include_fk,
                                quote_identifiers=cfg.quote_identifiers)
        if len(ddl) > cfg.max_schema_chars:
            ddl = ddl[: cfg.max_schema_chars] + "\n-- (schema truncated)"
        parts.append("### Database schema\n```sql\n" + ddl + "\n```")

    if cfg.include_evidence and evidence and evidence.strip():
        parts.append("### External knowledge\n" + evidence.strip())

    parts.append("### Question\n" + question.strip())
    parts.append(
        "### Answer\nWrite one SQLite query inside a ```sql block. "
        "Only the tables/columns above are available."
    )
    return "\n\n".join(parts)


def build_messages(question: str, schema: DbSchema, cfg: PromptConfig | None = None,
                   *, evidence: str | None = None) -> list[dict]:
    """返回 OpenAI chat 格式的 messages。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, schema, cfg, evidence=evidence)},
    ]


#: 多轮时的"上一轮反馈"消息模板。
#: 注意：{feedback} 里**绝不能**含 gold SQL / 正确答案，否则就是泄漏。
FEEDBACK_TEMPLATE = (
    "Your previous attempt:\n```sql\n{prev_sql}\n```\n\n"
    "{feedback}\n\n"
    "Write a corrected SQLite query in a ```sql block."
)


def build_feedback_message(prev_sql: str, feedback: str) -> dict:
    return {"role": "user", "content": FEEDBACK_TEMPLATE.format(prev_sql=prev_sql, feedback=feedback)}
