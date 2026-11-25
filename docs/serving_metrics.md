# 服务化指标：延迟、吞吐与成本

> 2026-11-12 · 数据来源：`results/baseline_bird_p6v3_*.json`（每条 Episode 都存了 `elapsed_sec`）
> 脚本：`scripts/latency_report.py`（离线统计）、`scripts/demo_agent.py`（单题演示）
> 目的：回答应用算法岗一定会问的三个问题——**单题多快？每千题多少钱？多轮贵多少？**

---

## 1. 一句话结论

**单轮：p50 2.44s / p95 4.09s，23.4 题/分钟，每千题 ¥1.78；三轮：p50 3.24s / p95 11.33s，每千题 ¥3.55。**
（单卡 4090、bf16、batch=1、HF `generate`，没有 vLLM/连续批处理）

---

## 2. 主表

| 结果 | n | 平均轮数 | p50 | p95 | p99 | 吞吐（题/分） | 每千题成本 | 正确率 |
|---|---|---|---|---|---|---|---|---|
| v3 单轮（全量） | 1534 | 1.00 | **2.44s** | **4.09s** | 5.60s | **23.4** | **¥1.78** | 50.85% |
| v3 单轮（307 抽样） | 307 | 1.00 | 2.41s | 4.08s | 5.08s | 23.5 | ¥1.77 | 51.79% |
| v3 三轮（307 抽样） | 307 | **1.92** | 3.24s | **11.33s** | 14.72s | 11.7 | **¥3.55** | 57.00% |

两个直接结论：

1. **三轮把 p95 拉高 2.8 倍**（4.09s → 11.33s）：因为要等模型做错、反馈、再生成；
   p50 只涨 33%（2.44 → 3.24），说明“大多数题第一轮就结束”，贵的是尾部；
2. **每千题成本翻倍**（¥1.78 → ¥3.55），但换来 +5.2pp 准确率——
   折合“每多对一个答案 ¥0.0027”，这个性价比可以在面试里讲。

延迟尾部：单轮 max 25.28s（1534 条里 p99 才 5.6s），来自个别生成过长 / 查询慢的题；
如果要做线上服务，**p99 才是 SLO 该盯的数**，不是均值。

---

## 3. 成本口径（别被问倒）

| 口径 | 怎么算 | 值（单轮） |
|---|---|---|
| **本地 GPU 机会成本** | 墙钟小时 × GPU 单价（¥2.5/卡时） | **¥1.78 / 千题** |
| 每个正确回答 | GPU 成本 / 正确数 | **¥0.0035** |
| DeepSeek API 对照 | prompt 684 tok + completion 40 tok，¥2/M + ¥8/M | ≈¥1.69 / 千题 |

**怎么讲这个对比**：本地单卡的成本和 DeepSeek API 是**同一量级**（¥1.8 vs ¥1.7/千题），
但本地模型准确率 50.85%、DeepSeek 84.33%——所以“本地跑”省的不是钱，是**数据不出域**和
**可微调/可迭代**；真要靠成本取胜，得上 vLLM + 批处理（见 §6）。

---

## 4. 单题演示（真实输出）

`scripts/demo_agent.py` 在本地 4060（4-bit、无 adapter）跑 BIRD 第 2 题，正好演示了完整闭环：

```
[问题] Please list the zip code of all the charter schools in Fresno County Office of Education.
[第 1 轮] 生成 4.15s
  SQL: SELECT DISTINCT s.Zip FROM frpm f JOIN schools s ON f.CDSCode = s.CDSCode
       WHERE f."Charter School (Y/N)" = 1 AND f."County Name" = 'Fresno'
  结果: 执行成功，34 行
[第 2 轮] 生成 3.47s
  SQL: ... AND f."County Name" = 'Fresno County'
  结果: 执行成功，0 行
  反馈: Execution succeeded: 0 rows. The result is EMPTY — double-check your WHERE / JOIN conditions.
        Value check: `Fresno County` is not found in column `County Name`.
        Closest existing values: 'Fresno'
[第 3 轮] 生成 3.49s
  SQL: ... AND f."County Name" = 'Fresno'     ← 改回来了
  结果: 执行成功，34 行
[判定] ✗ 错误（value_mismatch，gold 用的是 DISTRICT 过滤，不是 County）
[汇总] 3 轮 | 墙钟 11.16s | prompt 2365 tok / completion 116 tok
```

**这段演示的价值**：它同时展示了「执行成功 ≠ 正确」（34 行也可能是错的）、
「值检查能把拼错的值指出来」、「模型会在第 2 轮被反馈带偏、第 3 轮改回来」——
比任何 PPT 都有说服力。

换一道题复现同样的过程，把 `--index` 改成 0~7 任意一条即可（都在 california_schools 库上）。

```bash
python scripts/demo_agent.py --index 0 --turns 3 --value-check \
    --backend hf --model-path ../.models/Qwen3-4B-Instruct-2507 --load-in-4bit
```

---

## 5. 复现

```bash
# 1) 离线统计（不需要 GPU，只读 results/*.json）
python scripts/latency_report.py results/baseline_bird_p6v3_t1_full.json \
    results/baseline_bird_p6v3_t3_s5.json --gpu-price 2.5 --markdown

# 2) 单题演示（本地 8G 显卡，4-bit）
python scripts/demo_agent.py --index 2 --turns 3 --value-check \
    --backend hf --model-path ../.models/Qwen3-4B-Instruct-2507 --load-in-4bit

# 3) 用 vLLM / DeepSeek 等 OpenAI 兼容服务跑同一条题
python scripts/demo_agent.py --backend api --index 2 --turns 3
```

---

## 6. B-lite：vLLM 服务化并发（2026-11-16 实测）

**设置**：单卡 4090D 24G，Qwen3-4B 基座（v3 adapter 在旧实例上已不可得，吞吐结论与 adapter 无关），
BIRD dev 等距 220 条（stride 7），单轮，`--timeout 10 --max-rows 20000 --value-check`，
温度 0，**关闭 LLM 磁盘缓存**（否则第二档起全是缓存命中，p50 会假到 0.03s）。

| 配置 | n | 墙钟 | 吞吐（题/分） | p50 | p95 | p99 | EA | 加速 |
|---|---|---|---|---|---|---|---|---|
| HF 单卡串行 | 220 | 236s | **55.9** | 0.89s | 1.97s | 3.70s | 43.6% | 1.00x |
| vLLM `workers=1` | 220 | 142s | **93.1** | 0.53s | 1.16s | 2.12s | 43.2% | 1.67x |
| vLLM `workers=4` | 220 | 41s | **318.8** | 0.60s | 1.29s | 2.40s | 42.7% | 5.70x |
| vLLM `workers=8` | 220 | 34s | **391.7** | 0.65s | 1.41s | 2.52s | 42.7% | **7.01x** |
| vLLM `workers=16` | 220 | 32s | **415.1** | 0.73s | 1.70s | 3.97s | 43.2% | 7.42x |

三个可以讲的结论：

1. **单流也快 1.67 倍**：PagedAttention + CUDA graph/compile，p95 从 1.97s 降到 1.16s
   （不是靠堆并发换来的）；
2. **8 并发饱和**：8→16 只多 6%，而 p99 从 2.52s 涨到 3.97s——**甜点在 4~8**，
   再堆并发是用尾延迟换吞吐；
3. **每千题成本 ¥0.75 → ¥0.11**（6.8 倍降幅，GPU 口径），这是 continuous batching 的收益。

**一致性核对**：vLLM 与 HF 在同一批 220 条上**逐题判定一致 217/220（98.6%）**，
EA 43.64% vs 43.18%。3 条差异来自贪心解码的数值差异（不同 attention kernel 的浮点累加顺序），
不是模型/模板问题。

**边界**（面试别说错）：
- 这是**评测吞吐**，不是训练时的 rollout 吞吐；训练改成异步 vLLM 还要做权重同步 + staleness 控制（B-full）；
- adapter 不可得，用的是基座模型；吞吐与 adapter 无关，但 EA 不能和 v3 的 50.85% 直接比；
- 启动时间 84s 未摊进上表；长时间服务可忽略，按批跑要算进去。

**复现**（服务器，vLLM 0.33 + torch 2.14 独立 venv）：

```bash
# 1) 起服务（基座，无 LoRA）
vllm serve /root/autodl-tmp/qwen3-4b --served-model-name qwen3-4b \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000 &

# 2) 四档并发（脚本内部会关缓存）
python scripts/b_lite_throughput.py --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-4b --workers 1,4,8,16 --stride 7 --turns 1
```

**两个必须注意的点**：① 测吞吐时 `run_baseline` 必须加 `--no-cache`，否则第二档起全部命中
磁盘缓存（p50 会假到 0.03s）；② vLLM 0.33 的 `torch.compile` 依赖 `ninja`，
缺了会报 `FileNotFoundError: 'ninja'`，`pip install ninja` 即可。

---

## 7. B-full 第一步：rollout 采样基准（2026-11-16 实测）

训练每步 = 一条 prompt 的 G 条轨迹 + 一次更新。更新通常 <2s，所以瓶颈几乎全在 rollout。
这一节只测 rollout（同样的 agent：3 轮 / 判对才停 / value_check / T=1.0），
用 BIRD dev 的 25 条题 × G=4 做对照：

| 模式 | 题数×G | 墙钟 | 样本/秒 | 单步 rollout（group p50） | 相对 HF |
|---|---|---|---|---|---|
| **HF 串行**（`train_grpo` 现状） | 25×4 | 193.8s | **0.516** | **5.30s** | 1.00x |
| HF + 4 线程（对照） | 5×4 | 158.5s | 0.126 | 12.92s | **0.24x（更慢）** |
| vLLM 异步（G=4 并发） | 25×4 | 46.6s | **2.145** | **0.91s** | 4.16x / 5.8x |
| vLLM 异步（K=2，8 请求在飞） | 25×4 | 36.7s | 2.726 | 1.28s* | 5.28x |
| **vLLM 异步（K=4，16 请求在飞）** | 25×4 | 29.3s | **3.409** | **0.77s*** | **6.61x** |

\* K>1 时组的墙钟是“一批 K 条 prompt 的总时间 ÷ K”，即流水线摊薄后的等效单步时间。

**三条结论**：

1. **HF 多线程是负优化**：0.126 样本/秒，比串行慢 4 倍，GPU 利用率只有 18%——
   单卡 HF `generate` 在 Python 线程下互相抢锁，**必须换服务化**，不能靠开线程；
2. **vLLM 异步 4 并发就把单步 rollout 从 5.3s 压到 0.91s（5.8 倍）**，
   吞吐 0.516 → 2.145 样本/秒；
3. **多 prompt 在飞还能再提 1.6 倍**（3.409 样本/秒），但 group p95 从 3.8s 涨到 3.7s 左右、
   尾延迟变得不稳定——这就是为什么真实训练要加 **staleness cap**。

**折算到训练**：P6 的 24.8s/步里 rollout 约占 24s；如果异步 rollout 后按 K=4 的 0.77s/步算，
每步 ≈ 0.77（rollout）+ 1~2（更新）+ 权重同步开销，**目标是一个个位数秒的步**。

**B-full 第二步：异步训练 A/B（2026-11-17 完成）**

设置：SFT adapter（重训，732 步 / val_loss 0.156）起步，G=4、3 轮、30 步，
两臂同一个 SFT 起点、同一批 RL prompt（`rl_file=rl_train.jsonl`），max_len 都是 1536。

| 实验 | 每步 mean | p50 | p95 | rollout | update | sync | reward | zero_var | mask | EA（154 条） |
|---|---|---|---|---|---|---|---|---|---|---|
| 同步 HF（max_len 2560） | 22.13s | 15.5s | 59.7s | — | — | — | 0.667 | 0.733 | 0 | 57.14% |
| **同步 HF（max_len 1536）** | **22.35s** | **17.3s** | 63.6s | 20.77s | 1.55s | 0 | 0.675 | 0.667 | 0 | **57.79%** |
| **异步 vLLM + 热同步（1536）** | **11.65s** | **6.9s** | 29.7s | 9.62s | 1.65s | 0.33s | 0.717 | 0.767 | 0 | **57.14%** |

**结论**：

1. **每步 22.35s → 11.65s（1.92x）**，p50 从 17.3s 压到 **6.9s（2.5x）**；
2. **EA 没有掉**：57.79% vs 57.14%（差 0.65pp，n=154 的噪声内）；
3. **没有引入 off-policy 偏差**：`mask_mismatch` 恒 0、`ratio_mean` **全程 1.0**（每步同步权重）；
4. **LoRA 热同步只花 0.33s/步**（save 142MB + HTTP load），比“每步重启引擎”（84s）便宜两个数量级；
5. **rollout 仍是 9.6s**：因为多轮 + SQL 执行（timeout 10s）本身慢；p95 29.7s 说明尾部是真实存在的，
   下一步该做的是 **staleness cap + 预取**，而不是继续压平均值。

**踩过的三个坑（都会静默失败）**：

- vLLM 的动态 LoRA 接口默认关闭，必须 `export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`，
  否则 `POST /v1/load_lora_adapter` 返回 **404**；
- 单卡显存：vLLM 0.85 全量 20.8G 放不下；0.40 连 KV cache 都开不出来；
  最终 **0.46 + `--max-model-len 1536 --enforce-eager`**（11.3G）才能和 10.3G 的训练进程共存；
- `PATH` 里要有 `/root/vllm-env/bin`（`ninja` 在 venv 里，直接跑 `vllm` 不会自动加）。

---

## 8. 局限与下一步（面试会说“如果做线上”）

1. **batch=1，没有连续批处理**：23.4 题/分钟是“单请求串行”的数；
   vLLM 的 continuous batching 通常能再快 2~4 倍，且 p99 更稳——**B-lite 已实测：7.0 倍**；
2. **没有 KV cache 复用**：多轮之间 prompt 前缀重复度很高，线上应该做 prefix caching；
3. **没有超时降级**：p99 尾部靠 `--timeout 10` 兜底，线上应做分级超时 + 重试；
4. **成本只算 GPU**：没算 CPU/存储/运维，量级正确但别当成账单。

**下一步（B-full 第二步）**：rollout 已能异步（rollout 快 6.6 倍），剩下的是权重同步 + staleness；
这一步做完，训练的每步耗时才是真正被解决的。

---

*数据文件：`results/baseline_bird_p6v3_t1_full.json`、`..._t3_s5.json`、`..._t1_s5.json`；
B-lite：`results/baseline_blite_hf_s7.json`、`..._blite_vllm_w{1,4,8,16}.json`、`blite_vllm_summary.json`；
B-full 第一步：`results/bench_hf_sync.json`、`bench_hf_thread_small.json`、`bench_vllm_async_{c4,k2,k4}.json`。
GPU 单价按 ¥2.5/卡时（4090 行情）估算；换卡改 `--gpu-price` 即可。*
