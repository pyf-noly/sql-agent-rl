"""GRPO 的数学部分：组内优势、带 mask 的对数概率、裁剪目标、KL。

刻意把"数学"和"工程"分开：
- 组内优势用 **numpy** 实现，不需要 GPU 就能单测（``tests/test_grpo_math.py``）；
- 前向/反向才用 torch，且 torch 是**惰性导入**的，所以本模块在没有装 torch 的
  机器上也能 import（方便本地校验数据管线）。

GRPO 一句话版本
--------------
同一个问题采样 G 条轨迹，用**组内相对好坏**当优势::

    A_i = (r_i - mean(r)) / (std(r) + eps)

然后做 PPO 式的裁剪更新::

    ratio = exp(logp_new - logp_old)
    loss  = -mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )

为什么不用 value model：省一个和策略同规模的模型，7B 以下的实验里省显存比省样本更重要。

两个必须盯住的数
--------------
1. ``zero_var_ratio``：组内奖励全相同的比例。这部分样本**对梯度没有贡献**
   （优势全 0），比例太高说明任务太简单/太难，或采样温度太低；
2. ``clip_frac``：被裁剪的 token 比例。持续 >0.3 说明单步更新太猛，要降 lr。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch 只在真正训练时用到
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - 本地无 torch 环境
    torch = None
    F = None


# ---------------------------------------------------------------- 纯 numpy 部分

def group_advantages(rewards, group_size: int, *, eps: float = 1e-6,
                     normalize: bool = True) -> np.ndarray:
    """按组做标准化，返回与 ``rewards`` 等长的一维优势数组。

    Parameters
    ----------
    rewards:
        长度必须是 ``group_size`` 的整数倍，按"同一组的 G 条轨迹相邻"排列。
    normalize:
        True 时除以组内标准差（标准 GRPO）。False 时只减均值（Dr. GRPO 的做法，
        避免"全对/全错组"被除成噪声——两组都实现，P6 可以直接做消融）。
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    adv = r - r.mean(axis=1, keepdims=True)
    if normalize:
        std = r.std(axis=1, keepdims=True)
        # 组内全相同：优势直接置 0，别让 1e-6 把浮点噪声放大成信号
        adv = np.divide(adv, std + eps, out=np.zeros_like(adv), where=std > 1e-4)
    return adv.reshape(-1)


def group_stats(rewards, group_size: int) -> dict:
    """训练时要记录的组级统计（P4 验收项就在这里）。"""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    std = r.std(axis=1)
    return {
        "n_groups": int(r.shape[0]),
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "group_std_mean": float(std.mean()),
        "zero_var_ratio": float((std <= 1e-9).mean()),
        # 二分奖励下的两种"废组"：全对（没得学）和全错（没信号）
        "all_correct_ratio": float((r == 1.0).all(axis=1).mean()),
        "all_wrong_ratio": float((r == 0.0).all(axis=1).mean()),
        "n_distinct_rewards": int(len(np.unique(r))),
    }


# ------------------------------------------------- P6：轮级信用分配（纯 numpy）

def turn_advantages(first_correct_turn, n_turns, group_size: int, *,
                    beta: float = 1.0, gamma: float = 1.0,
                    eps: float = 1e-6) -> np.ndarray:
    """轮级优势矩阵 ``[G, T_max]``：把"哪一轮做对"拆成每一轮的信用。

    为什么需要它（P6 的核心动机）
    ---------------------------
    outcome-only GRPO 里，一组 G 条轨迹只要**最终**都对，episode 优势全是 0——
    可"第 1 轮就做对"和"第 3 轮才做对"显然不该同分。轮级优势就是把这部分
    被浪费掉的方差找回来：

    - ``y[i, t] = 1`` 当且仅当第 i 条轨迹**第一次做对**发生在第 t 轮（1-based）；
      最终没做对的轨迹，y 全 0；
    - 对每个轮次位置 t，在组内做标准化：``A_turn[i, t] = (y[i,t] - mean_t) / std_t``，
      组内该轮全同（std≈0）时置 0——和 outcome 同一套"没方差就不给梯度"的原则；
    - 再乘时间折扣 ``gamma**(t-1)``：越晚才对，同样的"做对了"给的优势越小。

    它**只做加法**：真正的更新量是 ``A_episode + beta * A_turn``（见
    :func:`combine_advantages`），所以 beta=0 时完全退化成 outcome-only，
    可以直接当消融的对照。

    Parameters
    ----------
    first_correct_turn:
        长度 G 的列表，元素是 1-based 的轮次；``0`` 表示最终没做对。
    n_turns:
        长度 G 的列表，每条轨迹实际跑了几轮（用于确定矩阵宽度）。
    group_size:
        必须等于 len(first_correct_turn)。
    """
    G = len(first_correct_turn)
    if G != group_size:
        raise ValueError(f"轨迹数 {G} 与 group_size {group_size} 不一致")
    T = max(1, max(n_turns) if n_turns else 1)
    y = np.zeros((G, T), dtype=np.float64)
    for i, fc in enumerate(first_correct_turn):
        if fc and 1 <= fc <= T:
            y[i, fc - 1] = 1.0

    adv = np.zeros_like(y)
    for t in range(T):
        col = y[:, t]
        sd = col.std()
        if sd <= 1e-6:          # 这一轮全对或全错：没有可学的相对信号
            continue
        adv[:, t] = (col - col.mean()) / sd
        adv[:, t] *= gamma ** t

    return adv * beta


def combine_advantages(episode_adv, turn_adv, beta: float = 1.0) -> np.ndarray:
    """轮级与整条轨迹的优势相加：``A[i, t] = A_ep[i] + beta * A_turn[i, t]``。

    beta=0 → 完全退化成 outcome-only（P5 的做法），用于消融。
    """
    ep = np.asarray(episode_adv, dtype=np.float64).reshape(-1, 1)
    return ep + beta * np.asarray(turn_adv, dtype=np.float64)


def fill_turn_advantages(episode_adv_i: float, turn_adv_row, spans, seq_len: int,
                         *, beta: float = 1.0) -> list[float]:
    """把一条轨迹的轮级优势铺到 token 序列上（返回长度 ``seq_len`` 的列表）。

    - ``spans`` 是 :class:`training.chatfmt.Encoded.turn_spans`（每个助手轮的 [s,e)）；
    - 第 t 轮覆盖的 token 取 ``A_ep + beta * A_turn[t]``；
    - 轮次之外的 token（prompt、反馈）取 0 —— 它们本来就在 loss_mask=0 区，
      填 0 只是让形状完整、并保证"如果 mask 意外为 1 也不会被错误加权"。
    """
    vec = [0.0] * int(seq_len)
    row = list(turn_adv_row) if turn_adv_row is not None else []
    for t, (s, e) in enumerate(spans):
        a = float(episode_adv_i) + (beta * float(row[t]) if t < len(row) else 0.0)
        for k in range(max(0, int(s)), min(int(e), len(vec))):
            vec[k] = a
    return vec


# ---------------------------------------------------------------- torch 部分

def _require_torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError("需要 torch：pip install torch（训练脚本只能在装好 torch 的机器上跑）")
    return torch


@dataclass
class Batch:
    """一组轨迹拼成的张量（右 padding）。

    ``advantages`` 既可以是 ``[B]``（整条轨迹一个优势，P4/P5 的做法），
    也可以是 ``[B, L]``（逐 token 优势，P6 的轮级信用分配用——不同轮次的
    token 带不同的优势）。:func:`grpo_loss` 两种都吃。
    """

    input_ids: object          # [B, L] long
    attention_mask: object     # [B, L] long
    loss_mask: object          # [B, L] float（1 = 参与 loss）
    advantages: object         # [B] 或 [B, L] float
    group_size: int = 0

    def to(self, device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            loss_mask=self.loss_mask.to(device),
            advantages=self.advantages.to(device),
            group_size=self.group_size,
        )

    @property
    def n_loss_tokens(self) -> int:
        return int(self.loss_mask.sum().item())


def pad_batch(encodings, advantages, pad_id: int, *, device=None, max_len: int | None = None):
    """把若干条 ``chatfmt.Encoded`` 右 padding 成张量，并附带优势。

    ``advantages`` 的元素可以是标量（整条轨迹一个优势）或与 ``input_ids``
    等长的序列（逐 token 优势，用于轮级信用分配）。
    """
    t = _require_torch()
    if len(encodings) != len(advantages):
        raise ValueError(f"轨迹数 {len(encodings)} 与优势数 {len(advantages)} 不一致")
    lens = [len(e.input_ids) for e in encodings]
    L = max(lens)
    if max_len:
        L = min(L, max_len) if max(lens) > max_len else L
    ids = t.full((len(encodings), L), pad_id, dtype=t.long)
    attn = t.zeros((len(encodings), L), dtype=t.long)
    lm = t.zeros((len(encodings), L), dtype=t.float)
    adv = t.zeros((len(encodings), L), dtype=t.float)
    for i, e in enumerate(encodings):
        n = min(len(e.input_ids), L)
        ids[i, :n] = t.tensor(e.input_ids[:n], dtype=t.long)
        attn[i, :n] = 1
        lm[i, :n] = t.tensor(e.loss_mask[:n], dtype=t.float)
        a = advantages[i]
        if isinstance(a, (int, float)):
            adv[i, :n] = float(a)
        else:
            seq = list(a)[:n]
            if seq:
                adv[i, :len(seq)] = t.tensor(seq, dtype=t.float)
    return Batch(input_ids=ids, attention_mask=attn, loss_mask=lm,
                 advantages=adv, group_size=len(encodings)).to(device) if device is not None \
        else Batch(input_ids=ids, attention_mask=attn, loss_mask=lm,
                   advantages=adv, group_size=len(encodings))


def sequence_logprobs(model, input_ids, attention_mask, loss_mask):
    """对给定序列做 teacher forcing，返回 (每个 token 的 logp, 平移后的 mask)。

    注意对齐：位置 t 的 logits 预测 t+1 的 token，所以返回长度是 ``L-1``。
    """
    t = _require_torch()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp, loss_mask[:, 1:]


def grpo_loss(policy_logp, old_logp, mask, advantages, *, clip_eps: float = 0.2):
    """PPO/GRPO 裁剪目标（token 级，按 mask 归一）。

    ``advantages`` 支持两种形状：
    - ``[B]``：整条轨迹一个优势（P4/P5 的 outcome-only）；
    - ``[B, L]``：逐 token 优势（P6 轮级信用分配：同一轨迹不同轮次带不同优势）。

    返回 ``(loss, metrics)``。loss 已包含负号，直接 ``backward()`` 即可。
    """
    t = _require_torch()
    if advantages.dim() == 2:
        # 逐 token 优势要与"平移后的 logp"对齐：位置 t 的 logits 预测 t+1，
        # 所以取 adv[:, 1:]（长度为 L-1，与 policy_logp 一致）。
        adv = advantages[:, 1:]
    else:
        adv = advantages.unsqueeze(-1)
    log_ratio = policy_logp - old_logp.detach()
    ratio = torch.exp(log_ratio)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)           # 越大越好 -> 取负做 minimize

    denom = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / denom

    with torch.no_grad():
        outside = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
        metrics = {
            "loss": float(loss.detach()),
            "ratio_mean": float(((ratio * mask).sum() / denom).detach()),
            "clip_frac": float(((outside * mask).sum() / denom).detach()),
            "approx_kl": float((((old_logp - policy_logp) * mask).sum() / denom).detach()),
            "n_loss_tokens": int(mask.sum().item()),
        }
    return loss, metrics


def k3_kl(policy_logp, ref_logp, mask):
    """k3 估计器：``exp(d) - d - 1``，``d = ref_logp - policy_logp``。

    比 ``-log(ratio)`` 方差小且恒非负，是 GRPO/PPO 里的常用做法。
    """
    t = _require_torch()
    d = ref_logp.detach() - policy_logp
    per_token = torch.exp(d) - d - 1.0
    denom = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / denom
