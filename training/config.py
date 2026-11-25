"""训练配置：从 JSON/YAML 读，字段名写错直接报错（不许静默用默认值）。

为什么不用 argparse 一把梭
--------------------------
P4~P6 会产生十几个实验（SFT / GRPO-outcome / GRPO-过程奖励 / 轮级优势...），
每个实验的参数必须能**原样存进结果目录**，否则三个月后没人能复现。
所以：配置文件进版本库，命令行只做少量覆盖。

为什么默认 JSON
--------------
少一个 PyYAML 依赖，且能在任何机器上校验；要 YAML 也支持（装了 pyyaml 就行）。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """模型 + LoRA。4B + LoRA 是 24G 卡上的"能跑且不亏"的配置。"""

    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dtype: str = "bfloat16"
    attn: str = "sdpa"
    """注意力实现：sdpa 是 PyTorch 自带，不需要编译 flash-attn。"""
    load_in_4bit: bool = False
    """4-bit（nf4）装载。给 8G 显存的本地卡（4060）跑 4B 推理/评测用；
    训练仍走 bf16 + LoRA（4-bit 训练要再配 QLoRA 的梯度设置，P4 没这需求）。"""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    grad_checkpointing: bool = True
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return _build(cls, d)


@dataclass
class CommonConfig:
    data_dir: str = "data/train"
    out_dir: str = "runs/default"
    seed: int = 42
    max_len: int = 2048
    logging_steps: int = 1
    save_every: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)


@dataclass
class SFTConfig(CommonConfig):
    """SFT 冷启动（E1）。"""

    out_dir: str = "runs/sft_qwen3_4b"
    epochs: float = 2.0
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    eval_every: int = 50
    repeat_multiturn: int = 3
    """多轮样本重复次数：P3 里多轮只占 6.3%，不重复就会被单轮样本淹掉。"""
    limit: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "SFTConfig":
        return _build(cls, d)


@dataclass
class GRPOConfig(CommonConfig):
    """GRPO 训练（P4 冒烟 / P5 主实验 / P6 BIRD）。"""

    out_dir: str = "runs/grpo_smoke"
    db_root: str = "data/raw/spider/database"
    """sqlite 根目录。BIRD 换成 ``data/raw/bird/train/train_databases``。"""
    prompt_preset: str = "spider"
    """``spider`` = 旧 prompt（逐字不变）；``bird`` = 带 evidence/类型/外键/引号。"""
    sql_max_rows: int = 1000
    """奖励判定时的结果行数上限。BIRD 要设 20000（有 55 条 gold 超过 1000 行）。"""
    compare: str = "rows"
    """奖励的等价判定口径：``rows``（Spider 历史口径）/ ``bird``（BIRD 官方集合比较）。
    必须与评测端一致，否则训练优化的目标和报告的数字不是同一个东西。"""
    value_check: bool = False
    """是否在"执行成功"时检查 WHERE 字面量是否存在（见 envs/value_check.py）。
    P6 第二轮起为 True：它把"取值拼错"这类语义错误变成可检测信号。"""
    eval_file: str = "data/processed/dev_sample.json"
    """``eval_every>0`` 时小样本评测用的数据文件。"""
    steps: int = 20
    group_size: int = 4
    """同一 prompt 采样几条。G=4 在 24G 上是"够用且跑得动"的折中。"""
    rl_file: str = "rl_train.jsonl"
    """RL prompt 文件名（在 data_dir 下）。P5 用校准过的 ``rl_p5_train.jsonl``。"""
    micro_batch: int = 2
    """一次前向放几条轨迹。G=8 时长序列的 logits 会吃掉 3~10G，
    拆成 2 条一组做梯度累积，显存曲线立刻变平（算法上完全等价）。"""
    lr: float = 1e-6
    kl_beta: float = 0.0
    """0 = 不做 KL 约束（P4 冒烟用）。P5 建议 0.02，防止策略跑飞。"""
    clip_eps: float = 0.2
    advantage_mode: str = "outcome"
    """``outcome`` = 整条轨迹一个优势（P4/P5）；``turn`` = 轮级信用分配（P6）。"""
    turn_beta: float = 1.0
    """轮级优势的权重。0 = 完全退化成 outcome-only（消融对照）。"""
    turn_gamma: float = 1.0
    """轮级时间折扣：越晚才做对，同样的"做对了"给的信用越小（1.0 = 不折扣）。"""
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    max_turns: int = 3
    sql_timeout: float = 2.0
    reward_correct: float = 1.0
    reward_wrong: float = 0.0
    reward_parse_error: float = -0.2
    """解析失败（没输出 ```sql 块）额外扣分：这是"格式都学不会"，不该和"答错"同分。"""
    reward_turn_penalty: float = 0.0
    """每多用一轮扣多少分（只在答对时生效）。模型准确率很高时，用它制造组内方差。"""
    reward_first_turn_bonus: float = 0.0
    """一轮就答对的额外奖励。比 turn_penalty 更激进，两者可叠加。"""
    eval_every: int = 0
    """>0 时每 N 步在 dev 子集上评一次（花钱花时间，冒烟阶段设 0）。"""
    eval_limit: int = 40
    limit: int = 0
    """只用前 N 条 RL prompt（冒烟）。"""

    @classmethod
    def from_dict(cls, d: dict) -> "GRPOConfig":
        return _build(cls, d)


def _build(cls, d: dict):
    """严格构造：多余字段报错，``model`` 支持嵌套 dict。"""
    if not isinstance(d, dict):
        raise TypeError(f"{cls.__name__} 需要 dict，拿到 {type(d).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"{cls.__name__} 不认识的配置项: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k == "model":
            kwargs[k] = v if isinstance(v, ModelConfig) else ModelConfig.from_dict(v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path | None, cls, overrides: dict | None = None):
    """读配置 + 命令行覆盖。``overrides`` 里值为 None 的项会被忽略。"""
    data: dict = {}
    if path:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("读 YAML 需要 pyyaml：pip install pyyaml") from e
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    merged = deep_merge(data, {k: v for k, v in (overrides or {}).items() if v is not None})
    return cls.from_dict(merged)


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
