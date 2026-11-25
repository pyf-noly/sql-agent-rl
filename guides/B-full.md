# B-full 执行手册：异步 rollout + LoRA 热同步

> 目标：把训练循环的 rollout 从「HF 进程内串行」换成「vLLM 服务异步」，
> 每步更新后把 LoRA 推进 vLLM（保持 on-policy），对比同步/异步的**每步耗时**与训练稳定性。
>
> 前置产物：SFT adapter（`runs/sft_bird_p6v3/adapter`）、训练库（15 个 RL 库）、vLLM 服务。

---

## 0. 一句话流程

```
SFT → 起 vLLM（--enable-lora，先挂 SFT adapter）→ tokenize 一致性检查
→ 同步臂（HF rollout）40 步 → 异步臂（vLLM rollout + 每步热同步）40 步
→ 两个 adapter 各评 150 条 → 比每步耗时 / ratio / EA
```

---

## 1. 起 vLLM（带 LoRA）

```bash
export PATH=/root/vllm-env/bin:$PATH
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # 不开这个，/v1/load_lora_adapter 会 404
cd /root/autodl-tmp/p6/sql-agent-rl
nohup vllm serve /root/autodl-tmp/qwen3-4b \
  --served-model-name qwen3-4b \
  --enable-lora --max-loras 2 --max-lora-rank 32 \
  --lora-modules sft=/root/autodl-tmp/p6/sql-agent-rl/runs/sft_bird_p6v3/adapter \
  --max-model-len 1536 --gpu-memory-utilization 0.46 --enforce-eager \
  --port 8000 > /root/vllm_b.log 2>&1 &
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 3; done
```

**为什么是 0.46 + 1536 + enforce-eager**（单卡和训练共存）：
- 0.85 全量占 20.8G，加训练 10.3G → 放不下；
- 0.40 连 KV cache 都开不出来（`No available memory for the cache blocks`）；
- 0.46 + 短 max_len + eager 占 11.3G，和 10.3G 的训练进程刚好共存（实测 30 步无 OOM）。

**常见坑**：
- `FileNotFoundError: 'ninja'` → `PATH` 里没有 `/root/vllm-env/bin`（直接执行 venv 的 `vllm` 不会自动加）；
- LoRA 加载失败 → 先去掉 `--enable-lora` 只服务基座，确认服务本身没问题，再加回来；
- `max-loras` 太小 → 热同步时会顶掉旧的；我们每步“先 load 新的、再 unload 旧的”，2 个槽够用。

---

## 2. 一致性检查（必做）

```bash
python scripts/check_vllm_template.py --base-url http://127.0.0.1:8000/v1 \
    --model sft --tokenizer /root/autodl-tmp/qwen3-4b
```

必须 5/5 OK。不一致就别往下跑——那说明 vLLM 的 chat template 和训练端
`apply_template` 不同，异步臂学的是另一种输入分布。

---

## 3. 同步臂（对照组）

```bash
python -m training.train_grpo \
  --config configs/train/grpo_bird_turn.json \
  --init-adapter runs/sft_bird_p6v3/adapter \
  --model /root/autodl-tmp/qwen3-4b \
  --steps 40 --out runs/grpo_sync40 > /root/grpo_sync40.log 2>&1
```

日志每步会打 `rollout Xs + update Ys + sync 0s = Zs`。

---

## 4. 异步臂（实验组）

```bash
python -m training.train_grpo \
  --config configs/train/grpo_bird_turn.json \
  --init-adapter runs/sft_bird_p6v3/adapter \
  --model /root/autodl-tmp/qwen3-4b \
  --rollout vllm --vllm-url http://127.0.0.1:8000/v1 --vllm-model sft \
  --sync hot \
  --steps 40 --out runs/grpo_async40 > /root/grpo_async40.log 2>&1
```

每步：`rollout`（G 条并发）→ `update`（HF 前向/反向）→ `sync`（save adapter + load 进 vLLM）。

**判读**：
- `ratio_mean` 应仍接近 1（每步同步 = 严格 on-policy）；
- `mask_mismatch` 必须 0；
- `sync_s` 是新增开销，预期 1~4s/步（142MB 落盘 + HTTP 加载）。

---

## 5. 评测两个 adapter

```bash
for arm in sync40 async40; do
  python scripts/run_baseline.py --dataset bird --backend hf \
    --adapter runs/grpo_${arm}/adapter \
    --model-path /root/autodl-tmp/qwen3-4b \
    --turns 1 --stride 10 --timeout 10 --value-check --tag ab_${arm}_t1
done
```

---

## 6. 成功判据

| 指标 | 期望 |
|---|---|
| 每步耗时 | 异步 < 同步的 1/2（rollout 5.3s → 0.9s，头） |
| `ratio_mean` | ≈1（每步同步） |
| `mask_mismatch` | 0 |
| EA（150 条） | 两臂差异在噪声内（±4pp） |
| 显存 | 异步臂训练端仍 ~12G（采样在 vLLM 进程） |

---

## 7. 面试口径

> 我把训练循环的 rollout 从进程内 HF 换成 vLLM 服务：G 条并发、每步把 LoRA 热加载进 vLLM
> 保持 on-policy。rollout 从 5.3s/步降到 0.9s，整步从 ~24.8s 降到个位数秒；
> 同时验证了 `mask_mismatch=0`、`ratio_mean≈1`，说明没有引入 off-policy 偏差。
> 如果每 N 步同步一次，就得到可控的 staleness —— 这是下一步要扫的参数。
