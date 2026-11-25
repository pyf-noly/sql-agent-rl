"""数据读取与批处理：把 ``data/train/*.jsonl`` 变成训练循环要的东西。

三件小事，但每件都能决定训练效不效
--------------------------------
1. **按长度分桶**：SQL 任务的序列长度从 300 到 2000 token 不等，随机组 batch
   会让每条样本都 padding 到批内最长，短样本白算一半算力。分桶后同批长度接近。
2. **多轮样本过采样**：P3 里多轮只占 6.3%，直接混训会被单轮淹没。按配置重复 N 次
   （重复的是"索引"，不是数据本体，省内存）。
3. **tokenize 一次缓存**：SFT 只有 2000 多条，启动时全编码进内存，
   训练时零 IO——在租来的机器上，省下的时间就是钱。
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from training.chatfmt import Encoded, encode_with_spans


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"找不到 {p}。先在仓库根目录跑：python scripts/build_p4_data.py")
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sft(path: str | Path, tokenizer, *, max_len: int,
             limit: int = 0) -> list[dict]:
    """读 SFT 数据并编码。返回 ``[{"enc": Encoded, "meta": dict}]``。

    支持 ``meta.mask_first_turn``：修复类样本的第一轮是**故意错的** SQL，
    如果对它也算 loss，模型会被训练成"先写个错的"——2026-11-05 的实测教训：
    多轮样本占比 49% 时，t1 从 51.5% 掉到 46.9%。所以这类样本只对**最后一轮**
    （改正后的 SQL）算 loss，第一轮当作条件上下文。
    """
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        meta = r.get("meta", {})
        enc = encode_with_spans(tokenizer, r["messages"], max_len=max_len)
        if meta.get("mask_first_turn") and len(enc.turn_spans) > 1:
            for s, e in enc.turn_spans[:-1]:
                for k in range(s, e):
                    enc.loss_mask[k] = 0
            enc.n_loss_tokens = sum(enc.loss_mask)
        out.append({"enc": enc, "meta": meta})
    return out


def load_rl(path: str | Path, *, limit: int = 0) -> list[dict]:
    rows = read_jsonl(path)
    return rows[:limit] if limit else rows


def expand_multiturn(items: list[dict], factor: int) -> list[int]:
    """返回训练用的索引序列：单轮出现 1 次，多轮出现 ``factor`` 次。"""
    if factor <= 1:
        return list(range(len(items)))
    idx: list[int] = []
    for i, it in enumerate(items):
        n_turns = it["meta"].get("n_turns", 1)
        idx.extend([i] * (factor if n_turns > 1 else 1))
    return idx


def bucketed_batches(lengths: list[int], batch_size: int, *, seed: int = 42,
                     shuffle: bool = True) -> list[list[int]]:
    """按长度排序后切块，再打乱批的顺序。

    批内长度接近 → padding 少；批顺序打乱 → 不会一直先学短样本。
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def length_summary(lengths: list[int]) -> dict:
    if not lengths:
        return {"n": 0}
    s = sorted(lengths)
    return {
        "n": len(s),
        "mean": round(statistics.mean(s), 1),
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
        "max": s[-1],
    }


def encoding_summary(encs: list[Encoded]) -> dict:
    """P4 验收要看的那几个数：长度分布 + 上下文占比 + mask 异常数。"""
    lens = [e.length for e in encs]
    return {
        "length": length_summary(lens),
        "n_loss_tokens": sum(e.n_loss_tokens for e in encs),
        "context_ratio_mean": round(statistics.mean([e.context_ratio for e in encs]), 4),
        "n_mismatch": sum(e.n_mismatch for e in encs),
        "n_truncated": sum(1 for e in encs if e.n_truncated),
    }
