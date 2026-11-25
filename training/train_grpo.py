#!/usr/bin/env python3
"""GRPO 训练（P4 冒烟 / P5 的 A 组：outcome-only 奖励）。

一条训练步在做什么（照着读代码就不会迷路）
----------------------------------------
1. 取一条 RL prompt（**没有答案**，只有问题和库）；
2. 同一 prompt 采样 G 条完整轨迹（G 条 = 一个"组"）；
3. 每条轨迹跑完多轮自修复，最后执行判对 → 奖励 1/0；
4. 组内标准化得到优势 A_i（这就是"相对好坏"，不需要 value model）；
5. 把 G 条轨迹拼成一个 batch，做一次裁剪策略梯度更新；
6. 记录：奖励均值、**零方差组比例**、loss、显存。

为什么这里 ratio 恒等于 1（不是 bug）
----------------------------------
每条样本只做**一次**更新就丢掉（1 个 inner epoch），所以算 loss 时权重没变过，
``exp(logp_new - logp_old) ≡ 1``，裁剪永远不会触发。这是 Dr.GRPO 的做法：
多轮 inner epoch 会让 ratio 漂移，反而需要重要性修正，收益不稳定。
裁剪分支保留是为了 P6——那时可能会做 2 个 inner epoch。

用法::

    # P4 冒烟（1.7B，20 步，只为"通"）
    python -m training.train_grpo --config configs/train/grpo_smoke_1p7b.json

    # 从 SFT 的 adapter 接着训（P5 正式实验）
    python -m training.train_grpo --config configs/train/grpo_qwen3_4b.json \
        --init-adapter runs/sft_qwen3_4b/adapter
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.agent import SqlAgent  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402
from rollout.prompt import BIRD_PROMPT  # noqa: E402
from training.chatfmt import encode_with_spans, messages_from_episode  # noqa: E402
from training.config import GRPOConfig, load_config  # noqa: E402
from training.data import encoding_summary, load_rl  # noqa: E402
from training.grpo import (combine_advantages, fill_turn_advantages,  # noqa: E402
                           group_advantages, group_stats, grpo_loss, k3_kl,
                           pad_batch, sequence_logprobs, turn_advantages)
from training.model import (adapter_disabled, count_params, grad_setup,  # noqa: E402
                            load_trainable_model, reset_vram_peak, vram_stats)
from training.rollout import HFPolicyClient, episode_reward, make_agent  # noqa: E402
from training.vllm_rollout import LoRASyncer, VLLMChatClient, check_server  # noqa: E402


def _prompt_cfg(preset: str):
    """``spider`` → 旧 prompt（None 即默认）；``bird`` → 带 evidence/类型/外键/引号。"""
    preset = (preset or "spider").lower()
    if preset == "spider":
        return None
    if preset == "bird":
        return BIRD_PROMPT
    raise ValueError(f"不认识的 prompt_preset: {preset}（可选 spider / bird）")


@torch.no_grad()
def quick_eval(model, tok, registry: DatabaseRegistry, rows: list[dict], *,
               limit: int, max_turns: int = 1, timeout: float = 2.0,
               prompt_cfg=None, max_rows: int = 1000,
               compare: str = "rows", value_check: bool = False) -> dict:
    """小样本贪心评测：回答"训练到底有没有让答案变好"这个唯一重要的问题。"""
    model.eval()
    client = HFPolicyClient(model, tok, max_new_tokens=256, temperature=0.0)
    ev = Evaluator(registry, timeout=timeout, max_rows=max_rows, compare=compare)
    agent = SqlAgent(client, registry, max_turns=max_turns, timeout=timeout,
                     temperature=0.0, stop_on_success=True,
                     prompt_cfg=prompt_cfg, max_rows=max_rows,
                     value_check=value_check)
    rows = rows[:limit]
    n_ok = 0
    for r in rows:
        ep = agent.run(r)
        if ep.final_sql:
            n_ok += int(ev.evaluate(r["db_id"], r["query"], ep.final_sql).correct)
    return {"n": len(rows), "ea": round(n_ok / max(1, len(rows)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/grpo_smoke_1p7b.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--init-adapter", default=None, help="从 SFT 的 adapter 起步")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rl-file", default=None, help="RL prompt 文件（在 data_dir 下）")
    ap.add_argument("--micro-batch", type=int, default=None, help="显存不够就降到 1")
    ap.add_argument("--max-len", type=int, default=None, help="序列上限，显存不够就降")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--kl-beta", type=float, default=None,
                    help="KL 系数。注意参考策略是**基座**（adapter 被禁用），"
                         "不是 SFT 后的策略，所以从 SFT adapter 起步时应当设 0")
    # ---- B-full：rollout 服务化 ----
    ap.add_argument("--rollout", choices=["hf", "vllm"], default="hf",
                    help="hf=进程内生成（原行为）；vllm=走 OpenAI 兼容服务，G 条并发")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--vllm-model", default="sft",
                    help="vLLM 启动时 --lora-modules 的名字（热同步后会自动切换）")
    ap.add_argument("--sync", choices=["none", "hot"], default="none",
                    help="hot=每次更新后把 LoRA 推进 vLLM（保持 on-policy）")
    ap.add_argument("--sync-every", type=int, default=1,
                    help="每 N 步同步一次（>1 会引入 staleness）")
    ap.add_argument("--parallel-prompts", type=int, default=1,
                    help="同时采样的 prompt 数（预留；当前一步仍只取 1 条）")
    args = ap.parse_args()

    cfg: GRPOConfig = load_config(args.config, GRPOConfig, {
        "out_dir": args.out, "steps": args.steps, "group_size": args.group_size,
        "limit": args.limit, "micro_batch": args.micro_batch,
        "max_len": args.max_len, "max_new_tokens": args.max_new_tokens,
        "rl_file": args.rl_file, "kl_beta": args.kl_beta,
        **({"model": {"name": args.model}} if args.model else {}),
    })
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out = ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"[配置] steps={cfg.steps} G={cfg.group_size} lr={cfg.lr} "
          f"kl_beta={cfg.kl_beta} max_turns={cfg.max_turns} T={cfg.temperature}")
    print(f"[模型] {cfg.model.name}  LoRA r={cfg.model.lora_r}  "
          f"init_adapter={args.init_adapter or '（无，从零开始）'}")

    model, tok = load_trainable_model(cfg.model.name, cfg.model, adapter=args.init_adapter)
    print(f"[参数] {count_params(model)}")
    print(f"[检查] {grad_setup(model)}  micro_batch={cfg.micro_batch} "
          f"max_len={cfg.max_len} max_new_tokens={cfg.max_new_tokens}")

    registry = DatabaseRegistry(ROOT / cfg.db_root)
    prompt_cfg = _prompt_cfg(cfg.prompt_preset)
    rl_path = ROOT / cfg.data_dir / cfg.rl_file
    rl = load_rl(rl_path, limit=cfg.limit)
    print(f"[数据] RL prompt {len(rl)} 条  ({cfg.rl_file})，"
          f"db_root={cfg.db_root}，prompt={cfg.prompt_preset}")
    if rl and "difficulty" in rl[0]:
        from collections import Counter
        print(f"[数据] 难度分布 {dict(Counter(r.get('difficulty') for r in rl))}")
    if rl and "p" in rl[0]:
        ps = [r["p"] for r in rl if "p" in r]
        print(f"[数据] 校准 p 均值 {sum(ps) / len(ps):.3f}（筛过的题应当在 0.2~0.8）")

    # 供 quick_eval 用：dev 子集（有 gold）
    dev_rows = []
    if cfg.eval_every:
        p = ROOT / cfg.eval_file
        if p.is_file():
            dev_rows = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[警告] eval_every={cfg.eval_every} 但找不到 {cfg.eval_file}，跳过小样本评测")

    # ---- B-full：vLLM 服务化 rollout 的初始化 ----
    current_model = args.vllm_model
    syncer = None
    if args.rollout == "vllm":
        info = check_server(args.vllm_url)
        ids = [m.get("id") for m in info.get("data", [])]
        print(f"[vllm] 服务就绪，可用模型 {ids}；初始模型={current_model}；sync={args.sync}")
        if args.sync == "hot":
            syncer = LoRASyncer(args.vllm_url)
            print("[vllm] LoRA 热同步已启用：每步 save_pretrained → /v1/load_lora_adapter")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    reset_vram_peak()

    t0 = time.time()
    t_tokens = 0
    t_rollouts = 0
    last = {}
    order: list[int] = []

    for step in range(cfg.steps):
        # 打乱顺序遍历，走完一轮重新洗（避免一直学同一批题、也避免固定顺序的偏差）
        if not order:
            order = list(range(len(rl)))
            random.Random(cfg.seed + step).shuffle(order)
        sample = rl[order.pop()]
        rollout_t0 = time.perf_counter()

        if args.rollout == "vllm":
            # 服务化采样：G 条轨迹并发请求 vLLM；agent 复用同一个 HTTP 客户端
            client = VLLMChatClient(args.vllm_url, current_model,
                                    temperature=cfg.temperature,
                                    max_tokens=cfg.max_new_tokens)
            agent, ev = make_agent(
                client, registry, max_turns=cfg.max_turns,
                timeout=cfg.sql_timeout, temperature=cfg.temperature,
                prompt_cfg=prompt_cfg, max_rows=cfg.sql_max_rows,
                compare=cfg.compare, value_check=cfg.value_check,
            )

            def _one(g: int):
                torch.manual_seed(cfg.seed * 1_000_003 + step * 101 + g)
                return agent.run(sample)

            with ThreadPoolExecutor(max_workers=cfg.group_size) as pool:
                episodes = list(pool.map(_one, range(cfg.group_size)))
            rewards, infos = [], []
            for ep in episodes:
                r, info = episode_reward(
                    ep, sample, ev,
                    correct=cfg.reward_correct, wrong=cfg.reward_wrong,
                    parse_error=cfg.reward_parse_error,
                    turn_penalty=cfg.reward_turn_penalty,
                    first_turn_bonus=cfg.reward_first_turn_bonus,
                )
                rewards.append(r)
                infos.append(info)
        else:
            client = HFPolicyClient(
                model, tok, max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature, top_p=cfg.top_p,
            )
            agent, ev = make_agent(
                client, registry, max_turns=cfg.max_turns,
                timeout=cfg.sql_timeout, temperature=cfg.temperature,
                prompt_cfg=prompt_cfg, max_rows=cfg.sql_max_rows,
                compare=cfg.compare, value_check=cfg.value_check,
            )

            episodes, rewards, infos = [], [], []
            for g in range(cfg.group_size):
                # 每条样本单独设种子：可复现，且同组内不会复用同一串随机数
                torch.manual_seed(cfg.seed * 1_000_003 + step * 101 + g)
                ep = agent.run(sample)
                r, info = episode_reward(
                    ep, sample, ev,
                    correct=cfg.reward_correct, wrong=cfg.reward_wrong,
                    parse_error=cfg.reward_parse_error,
                    turn_penalty=cfg.reward_turn_penalty,
                    first_turn_bonus=cfg.reward_first_turn_bonus,
                )
                episodes.append(ep)
                rewards.append(r)
                infos.append(info)
        rollout_s = time.perf_counter() - rollout_t0

        adv = group_advantages(rewards, cfg.group_size)
        schema = registry.schema(sample["db_id"])
        # 注意 prompt_cfg：BIRD 的 prompt 带 evidence/类型/引号，训练序列必须与
        # rollout 时**逐字一致**，否则学的是另一种输入分布。
        encs = [encode_with_spans(tok, messages_from_episode(ep, schema, prompt_cfg),
                                  max_len=cfg.max_len) for ep in episodes]

        # ---- P6：轮级信用分配 ----
        # outcome-only 时 adv 是 [G]（整条轨迹一个优势）；turn 模式下把它铺成
        # 逐 token 的优势：每条轨迹里，第 t 轮的那些 token 用 A_ep + beta * A_turn[i,t]。
        adv_batch: list = list(adv)
        turn_info: dict = {}
        if cfg.advantage_mode == "turn":
            first_correct = [next((k + 1 for k, t in enumerate(ep.turns) if t.correct), 0)
                             for ep in episodes]
            n_turns = [ep.n_turns for ep in episodes]
            t_adv = turn_advantages(first_correct, n_turns, cfg.group_size,
                                    beta=1.0, gamma=cfg.turn_gamma)
            adv_batch = []
            for i, enc in enumerate(encs):
                adv_batch.append(fill_turn_advantages(
                    float(adv[i]), t_adv[i], enc.turn_spans, len(enc.input_ids),
                    beta=cfg.turn_beta))
            turn_info = {
                "first_correct_ratio": round(sum(1 for x in first_correct if x) / len(first_correct), 4),
                "turn_adv_abs_mean": round(float(np.abs(t_adv).mean()), 4),
                "turn_adv_nonzero": int((np.abs(t_adv) > 1e-9).sum()),
            }

        # 更新阶段必须切回 train()：梯度检查点只在训练模式下生效，
        # 停在 eval() 会把整段前向的激活全留在显存里（4B×G 条直接 OOM）。
        # 采样时的 eval() 由 HFPolicyClient.chat 自己负责。
        update_t0 = time.perf_counter()
        model.train()
        total_tokens = max(1, sum(sum(e.loss_mask) for e in encs))
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        metrics: dict = {}
        kl_vals: list[float] = []

        # 逐微批前向 + 梯度累积：等价于整组一次前向，但显存峰值只按 micro_batch 算
        mb = max(1, cfg.micro_batch)
        for i in range(0, len(encs), mb):
            batch = pad_batch(encs[i:i + mb], adv_batch[i:i + mb], tok.pad_token_id,
                              device=model.device)
            logp, loss_mask = sequence_logprobs(
                model, batch.input_ids, batch.attention_mask, batch.loss_mask)
            old_logp = logp.detach()          # 单步更新：权重没动过，ratio 恒为 1
            loss, m = grpo_loss(logp, old_logp, loss_mask, batch.advantages,
                                clip_eps=cfg.clip_eps)

            if cfg.kl_beta > 0:
                with torch.no_grad(), adapter_disabled(model):
                    ref_logp, _ = sequence_logprobs(
                        model, batch.input_ids, batch.attention_mask, batch.loss_mask)
                kl = k3_kl(logp, ref_logp, loss_mask)
                loss = loss + cfg.kl_beta * kl
                kl_vals.append(float(kl.detach()))

            # 按 token 数加权，保证与"整组一次算"的 loss 数值一致
            weight = m["n_loss_tokens"] / total_tokens
            (loss * weight).backward()
            loss_sum += m["loss"] * weight
            metrics = m

        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        update_s = time.perf_counter() - update_t0

        # ---- B-full：把刚更新完的 LoRA 推进 vLLM（保持下一步 on-policy）----
        sync_s = 0.0
        if args.rollout == "vllm" and args.sync == "hot" and (step + 1) % max(1, args.sync_every) == 0:
            sync_t0 = time.perf_counter()
            sync_dir = out / "sync_tmp"
            sync_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(sync_dir)
            name = f"grpo_s{step + 1}"
            syncer.push(name, str(sync_dir))
            current_model = name
            sync_s = time.perf_counter() - sync_t0

        t_rollouts += len(episodes)
        t_tokens += sum(len(e.input_ids) for e in encs)
        stats = group_stats(rewards, cfg.group_size)
        enc_stat = encoding_summary(encs)
        kl_val = sum(kl_vals) / len(kl_vals) if kl_vals else None

        rec = {
            "step": step + 1,
            "difficulty": sample.get("difficulty"),
            "p_calibrated": sample.get("p"),
            "reward_mean": round(stats["reward_mean"], 4),
            "zero_var_ratio": round(stats["zero_var_ratio"], 4),
            "all_correct_ratio": round(stats["all_correct_ratio"], 4),
            "all_wrong_ratio": round(stats["all_wrong_ratio"], 4),
            "turns_mean": round(sum(i["n_turns"] for i in infos) / len(infos), 2),
            "parse_error": sum(1 for i in infos if i["reason"] == "parse_error"),
            "loss": round(loss_sum, 4),
            "ratio_mean": round(metrics["ratio_mean"], 4),
            "clip_frac": round(metrics["clip_frac"], 4),
            "grad_norm": round(grad_norm, 3),
            "kl": None if kl_val is None else round(kl_val, 5),
            "n_loss_tokens": total_tokens,
            "context_ratio": enc_stat["context_ratio_mean"],
            "mask_mismatch": enc_stat["n_mismatch"],
            "vram_gb": vram_stats().get("peak_alloc_gb"),
            "elapsed_s": round(time.time() - t0, 1),
            "rollout_s": round(rollout_s, 2),
            "update_s": round(update_s, 2),
            "sync_s": round(sync_s, 2),
            "step_s": round(rollout_s + update_s + sync_s, 2),
        }
        if turn_info:
            rec.update(turn_info)
        last = rec
        print(f"[{rec['step']:>3}/{cfg.steps}] reward {rec['reward_mean']:.2f} "
              f"零方差 {rec['zero_var_ratio']:.0%} | loss {rec['loss']:.4f} "
              f"| 轮数 {rec['turns_mean']} | 显存 {rec['vram_gb']}G | {rec['elapsed_s']}s"
              f" | rollout {rec['rollout_s']}s + update {rec['update_s']}s"
              f" + sync {rec['sync_s']}s = {rec['step_s']}s"
              + (f" | 轮级 adv 非零 {rec.get('turn_adv_nonzero')}" if turn_info else ""))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            model.save_pretrained(out / f"adapter_step{step + 1}")
            print(f"        [保存] adapter_step{step + 1}")

        if cfg.eval_every and dev_rows and (step + 1) % cfg.eval_every == 0:
            qe = quick_eval(model, tok, registry, dev_rows, limit=cfg.eval_limit,
                            max_turns=cfg.max_turns, timeout=cfg.sql_timeout,
                            prompt_cfg=prompt_cfg, max_rows=cfg.sql_max_rows,
                            compare=cfg.compare, value_check=cfg.value_check)
            print(f"        [eval] EA={qe['ea']:.2%} (n={qe['n']})")
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step + 1, **qe}, ensure_ascii=False) + "\n")

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    summary = {
        "stage": "grpo",
        "model": cfg.model.name,
        "init_adapter": args.init_adapter,
        "steps": cfg.steps,
        "group_size": cfg.group_size,
        "reward": {"correct": cfg.reward_correct, "wrong": cfg.reward_wrong,
                   "parse_error": cfg.reward_parse_error},
        "rollouts": t_rollouts,
        "tokens": t_tokens,
        "elapsed_s": round(time.time() - t0, 1),
        "vram": vram_stats(),
        "last": last,
        "config": {k: v for k, v in cfg.__dict__.items() if k != "model"},
        "adapter": str((out / "adapter").relative_to(ROOT)),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\n[done] {cfg.steps} 步，{t_rollouts} 条轨迹，耗时 {summary['elapsed_s']}s")
    print(f"[done] 显存峰值 {summary['vram'].get('peak_alloc_gb')}G，adapter 在 {summary['adapter']}")
    print("[验收] 上面每行的 mask_mismatch 必须是 0；zero_var_ratio 超过 0.8 说明奖励没区分度")
    return 0


if __name__ == "__main__":
    sys.exit(main())
