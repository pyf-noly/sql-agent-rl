#!/usr/bin/env python3
"""SFT 冷启动（实验 E1）：用 P3 的轨迹把 4B 教会"按格式写 SQL"。

这一步在 P4 里不是为了刷分，而是为了**给 GRPO 一个像样的起点**：
基座模型如果连 ```sql 格式都稳不住，GRPO 的奖励会大面积是 0，组内没有方差，
梯度恒为 0——训练看起来"在跑"，其实什么都没学。

用法::

    python -m training.train_sft --config configs/train/sft_qwen3_4b.json
    python -m training.train_sft --config ... --limit 100      # 冒烟
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.config import SFTConfig, load_config  # noqa: E402
from training.data import (bucketed_batches, encoding_summary, expand_multiturn,  # noqa: E402
                           load_sft)
from training.model import (count_params, load_trainable_model, reset_vram_peak,  # noqa: E402
                            vram_stats)


def _make_batch(items: list[dict], pad_id: int, device) -> tuple:
    """右 padding 成张量。SFT 不需要优势，只要 ids/attention/loss_mask。"""
    encs = [it["enc"] for it in items]
    L = max(e.length for e in encs)
    ids = torch.full((len(encs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(encs), L), dtype=torch.long)
    mask = torch.zeros((len(encs), L), dtype=torch.float)
    for i, e in enumerate(encs):
        n = e.length
        ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attn[i, :n] = 1
        mask[i, :n] = torch.tensor(e.loss_mask, dtype=torch.float)
    return ids.to(device), attn.to(device), mask.to(device)


def masked_lm_loss(model, ids, attn, mask) -> torch.Tensor:
    """只在助手 token 上算交叉熵。

    为什么不直接用 ``labels=ids``：那会把 schema 和工具反馈也算进 loss，
    模型会去学"复述表名"，而不是"写 SQL"。
    """
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    targets = ids[:, 1:]
    loss_tok = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    m = mask[:, 1:]
    return (loss_tok * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def eval_loss(model, items: list[dict], pad_id: int, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(items), 4):
        batch = items[i:i + 4]
        ids, attn, mask = _make_batch(batch, pad_id, device)
        loss = masked_lm_loss(model, ids, attn, mask)
        total += float(loss) * len(batch)
        n += len(batch)
    model.train()
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/sft_qwen3_4b.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条（冒烟用）")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg: SFTConfig = load_config(args.config, SFTConfig, {
        "out_dir": args.out, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size,
        **({"model": {"name": args.model}} if args.model else {}),
        **({"limit": args.limit} if args.limit else {}),
    })
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out = ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"[配置] {json.dumps({k: v for k, v in cfg.__dict__.items() if k != 'model'}, ensure_ascii=False)}")
    print(f"[模型] {cfg.model.name}  LoRA r={cfg.model.lora_r}")

    model, tok = load_trainable_model(cfg.model.name, cfg.model)
    stats = count_params(model)
    print(f"[参数] 总 {stats['params_total_b']}B，可训练 {stats['params_trainable_m']}M "
          f"({stats['trainable_ratio']:.2%})")

    data_dir = ROOT / cfg.data_dir
    train = load_sft(data_dir / "sft_train.jsonl", tok, max_len=cfg.max_len, limit=cfg.limit)
    val = load_sft(data_dir / "sft_val.jsonl", tok, max_len=cfg.max_len,
                   limit=max(20, cfg.limit // 5) if cfg.limit else 0)
    print(f"[数据] train {len(train)} 条 / val {len(val)} 条")
    print(f"[数据] {encoding_summary([x['enc'] for x in train])}")
    bad = sum(x["enc"].n_mismatch for x in train)
    if bad:
        print(f"[警告] mask 与原文对不上 {bad} 处——先去查 chat 模板，别急着训练")

    idx = expand_multiturn(train, cfg.repeat_multiturn)
    lengths = [train[i]["enc"].length for i in idx]
    print(f"[数据] 多轮过采样后 {len(idx)} 条（{cfg.repeat_multiturn}x 多轮）")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 一个 epoch 的"优化步数" = 批数 / 梯度累积
    batches_per_epoch = math.ceil(len(idx) / cfg.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / cfg.grad_accum)
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    reset_vram_peak()
    device = model.device
    pad_id = tok.pad_token_id
    model.train()

    t0 = time.time()
    step, seen = 0, 0
    running: list[float] = []
    log_every = max(1, cfg.logging_steps)

    for epoch in range(int(math.ceil(cfg.epochs))):
        batches = bucketed_batches(lengths, cfg.batch_size, seed=cfg.seed + epoch)
        for bi, batch_pos in enumerate(batches):
            if step >= total_steps:
                break
            items = [train[idx[p]] for p in batch_pos]
            ids, attn, mask = _make_batch(items, pad_id, device)
            loss = masked_lm_loss(model, ids, attn, mask)
            (loss / cfg.grad_accum).backward()

            running.append(float(loss))
            seen += len(items)

            if (bi + 1) % cfg.grad_accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    rec = {
                        "step": step, "epoch": epoch + 1,
                        "loss": round(sum(running) / len(running), 4),
                        "lr": f"{lr_at(step):.2e}",
                        "seen": seen,
                        "vram_gb": vram_stats().get("peak_alloc_gb"),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    running.clear()
                    print(f"[{step:>5}/{total_steps}] loss {rec['loss']:.4f}  "
                          f"lr {rec['lr']}  显存 {rec['vram_gb']}G  {rec['elapsed_s']}s")
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if cfg.eval_every and step % cfg.eval_every == 0:
                    vl = eval_loss(model, val, pad_id, device)
                    print(f"        val_loss {vl:.4f}")
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "val_loss": round(vl, 4)}) + "\n")
            if step >= total_steps:
                break

    final_val = eval_loss(model, val, pad_id, device)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    summary = {
        "stage": "sft",
        "model": cfg.model.name,
        "lora": {"r": cfg.model.lora_r, "alpha": cfg.model.lora_alpha},
        "n_train": len(train), "n_val": len(val),
        "steps": step, "epochs": cfg.epochs,
        "final_val_loss": round(final_val, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "vram": vram_stats(),
        "config": {k: v for k, v in cfg.__dict__.items() if k != "model"},
        "adapter": str((out / "adapter").relative_to(ROOT)),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\n[done] val_loss {final_val:.4f}，adapter 存到 {summary['adapter']}")
    print(f"[done] 耗时 {summary['elapsed_s']}s，显存峰值 {summary['vram'].get('peak_alloc_gb')}G")
    return 0


if __name__ == "__main__":
    sys.exit(main())
