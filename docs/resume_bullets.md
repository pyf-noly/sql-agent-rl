# 简历三行（定稿）与追问口径

> 2026-11-06 · 数字全部来自 `results/*.json`（见 `docs/p7_statistics.md`）
> 原则：**写机制、写可复现的数字、写清样本量**；不写"显著提升"（不成立）。

---

## 1. 定稿版（三行，中文简历）

> **面向 Text-to-SQL 的多轮自我修复 Agent：环境、数据与轮级信用分配**
> （个人项目，开源；单卡 4090 完成全流程，端到端可复现，约 ¥58）
>
> - **环境与数据**：搭只读 SQLite 多轮环境（`mode=ro` + 单语句白名单 + 2s 软超时 +
>   9 类结构化反馈 + 取值存在性检查，230 个单测），合成 3918 条 BIRD 多轮修复数据
>   （schema 错误 1133 / 取值错误 795，成功率 94% / 99%）；**库级切分 54 SFT / 15 RL /
>   11 dev 零重叠**作为未见库泛化协议；Qwen3-4B + LoRA 经 SFT/GRPO，Spider dev300
>   单轮 78.00% → 三轮 87.00%；BIRD dev 全量 1534 条单轮 **50.85%**、
>   307 条同批上三轮自修复 **57.00%**；单轮 p50 2.44s / p95 4.09s、每千题 ¥1.78（单卡 4090）；
> - **轮级信用分配（核心）**：定位 outcome-only GRPO 在多轮任务上的信用分配缺口
>   （换 BIRD 把零方差 0.83 降到 0.615，**有梯度步数 13~19% → 38.5%**），实现轮级组内优势
>   （`A[i,t] = A_ep + β·γ^(t−1)·normalize(首次做对的轮次)`，β=0 严格退化为 outcome-only）；
>   同等训练预算下单轮 EA **52.77% vs 51.14%**（+1.63pp，hard 档 +4.3pp，n=307 单 seed）；
> - **机制归因**：失败分析证明执行反馈只能发现一小部分错误（首轮错题 79%~90% 是
>   "能执行但结果错"；**Spider-SFT 迁 BIRD 仅 +1.4pp（n=74），故从零 SFT**）；
>   加入值检查后"多轮才救回"从 2.6% 升到 **5.2%**，三轮 EA +2.93pp；并证明多轮 SFT
>   若不 mask "故意写错的第一轮"，单轮会掉 4.6pp（mask 后回到 51.79%，307 口径）。

### 1.1 应用算法岗变体（版面允许时，在第三行后加这条）

> - **服务化**：单卡 4090 / batch=1 下单轮 p50 **2.44s**、p95 **4.09s**（23.4 题/分），
>   三轮 p95 11.3s；每千题成本 **¥1.78**（折合每多对一个答案 ¥0.0027），与 DeepSeek API
>   （≈¥1.69/千题）同量级，差别在数据不出域与可迭代；换 vLLM 服务化后评测吞吐
>   **55.9 → 391.7 题/分（8 并发，7.0x）**、每千题 ¥0.11，与 HF 逐题判定一致 98.6%；
>   **异步 rollout**（16 请求在飞）把训练采样从 **0.52 提到 3.41 样本/秒（6.6x）**、
>   单步 rollout 5.3s → 0.77s（HF 多线程反而 0.13，验证必须服务化）；**端到端训练**
>   每步 **22.35s → 11.65s（1.92x，p50 2.5x）**，EA 57.79% vs 57.14%（噪声内），
>   `mask_mismatch=0`、`ratio_mean≡1`（每步 LoRA 热同步，严格 on-policy，同步仅 0.33s/步）；
>   提供 CLI 单题演示（`scripts/demo_agent.py`）与延迟/成本统计（`scripts/latency_report.py`），可现场跑。

---

## 2. 更短的两行版（版面紧张时）

> - 搭多轮 Text-to-SQL 自我修复环境与 3918 条修复数据（230 单测），4B + LoRA 经 SFT/GRPO：
>   Spider dev300 单轮 78.00% → 三轮 87.00%；BIRD dev 全量单轮 50.85%、三轮 57.00%；
> - 实现轮级信用分配（β=0 可严格退化为 outcome-only），单轮 52.77% vs 51.14%、
>   hard +4.3pp；定位"反馈信息量决定自修复上限"（值检查使修复率 2.6%→5.2%）。

---

## 3. 英文版

> **Multi-turn Self-Repairing Text-to-SQL Agent: Environment, Data, and Turn-level Credit Assignment**
> (personal project, open source)
>
> - Built a read-only SQLite multi-turn environment (read-only URI, single-statement whitelist,
>   2s soft timeout, 9-class structured feedback, value-existence check; 230 unit tests) and
>   synthesized 3,918 multi-turn repair samples for BIRD (94%/99% success); Qwen3-4B + LoRA with
>   SFT/GRPO reached 78.00% → 87.00% on Spider dev300 (1-turn → 3-turn), **50.85%** on the full
>   BIRD dev (1,534) and **57.00%** 3-turn on the 307-question subset;
> - Identified the credit-assignment gap of outcome-only GRPO for multi-turn agents (measured
>   zero-variance ratio 0.615) and implemented turn-level group-relative advantages
>   (`A[i,t] = A_ep + β·γ^(t−1)·normalize(first-correct turn)`, β=0 reduces exactly to
>   outcome-only): **52.77% vs 51.14%** at equal budget (+4.3pp on hard, n=307, single seed);
> - Showed by failure analysis that execution feedback can only detect 10–21% of errors
>   (79–90% of first-turn errors execute but return wrong results); adding a
>   value-existence check raised the repair rate from 2.6% to **5.2%** and 3-turn EA by 2.93pp;
>   also showed that not masking the deliberately-broken first turn costs 4.6pp on 1-turn EA.

---

## 4. 每一行的证据（被追问时直接给）

| 简历里的说法 | 证据文件 | 追问时的回答 |
|---|---|---|
| 230 个单测 | `pytest tests -q` → 230 passed | 沙箱 40、错误分类 29、评测 25、其余按模块；CI 里不依赖 GPU |
| 零方差 0.615 | `docs/p6_results.md` §3.2 | 200 步里 67 步全对、56 步全错、77 步可学（38.5%）；Spider 是 0.74~0.90 |
| 52.77% vs 51.14% | `results/baseline_bird_p6_turn_t1_s5.json` vs `..._b0_t1_s5.json` | 同批 307 条、同 120 步、只改 advantage；配对 p=0.18，不显著 |
| 57.00% | `results/baseline_bird_p6v3_t3_s5.json` | 307 条 stride 5、BIRD 官方 set 口径、`--until-correct` |
| 50.85% | `results/baseline_bird_p6v3_t1_full.json` | 全量 1534 条、官方口径，±1.28pp；307 子集逐条一致 |
| +2.93pp | `scripts/paired_stats.py` | v3 t3 vs v1 t3，31/22 不一致对，p=0.272 |
| 修复率 2.6% → 5.2% | 同两个 JSON 的 `n_turns` 分布 | "多轮才救回"= 最终对且轮数 >1；v1 8 条、v3 16 条 |
| p50 2.44s / p95 4.09s | `results/baseline_bird_p6v3_t1_full.json` 的逐条 `elapsed_sec` | `scripts/latency_report.py` 现场算；单卡 4090、bf16、batch=1 |
| 每千题 ¥1.78 | 同上：墙钟 65.4 min × ¥2.5/卡时 ÷ 1534 × 1000 | 成本口径是 GPU 机会成本；API 对照 ≈¥1.69/千题（DeepSeek 价目） |
| 391.7 题/分（7.0x） | `results/blite_vllm_summary.json`、`results/baseline_blite_vllm_w8.json` | 220 条 stride 7、4090D、8 并发、关缓存；与 HF 逐题一致 217/220 |
| 3.41 样本/秒（6.6x） | `results/bench_vllm_async_k4.json`、`results/bench_hf_sync.json` | 25 题×G=4、3 轮、同 agent；单步 rollout 5.30s → 0.77s |
| 22.35s → 11.65s（1.92x） | `runs/grpo_sync30_1536/metrics.jsonl`、`runs/grpo_async30/metrics.jsonl` | 30 步同起点；p50 17.3→6.9s；EA 57.79% vs 57.14%（154 条）；mask=0、ratio≡1 |

---

## 5. 不能写的说法（会被一问就穿）

| ❌ 不能写 | 为什么 | ✅ 改写成 |
|---|---|---|
| "轮级优势显著提升 1.63pp" | p=0.18，不显著 | "方向一致的提升（+1.63pp），n=307 未达显著" |
| "BIRD 上比 SFT 提升 2.93pp（SOTA 对比）" | 是相对上一代**自己的**数据，不是相对 SOTA | "相对上一代多轮数据 +2.93pp；BIRD 官方论文的 4B 基线在 50% 量级" |
| "EA 87%" | 是 Spider dev300 三轮口径，不是 test 集 | 写明"dev300 / 三轮 / 未调参污染" |
| "用 verl 训练" | 实际是自研轻量 GRPO | "自研 GRPO（约 1400 行），单卡可跑；不依赖 verl/vLLM" |
| "多轮提升了 5.21pp，模型学会用反馈" | 多轮增益主要来自"多给两次机会"，不是"学会了反馈" | "多轮给了 +5.21pp（p<0.001），但 153 条错题只有 16 条被后续轮救回，修复能力仍是瓶颈" |

---

## 6. 三行背后的取舍（如果面试官问"你最想讲哪一行"）

**第二行。** 因为第一行是工程量，第三行是分析，第二行是"发现问题 + 设计机制 + 做消融"：

- 问题不是"GRPO 调参"，而是"多轮任务里 outcome reward 的信用分配分辨率不够"；
- 机制不是拍脑袋：先量化了零方差（0.615），再定位"全对组里过程分化的方差被浪费"，
  才设计 `A_ep + A_turn`；
- 消融是免费的（β=0 严格退化），所以"轮级 vs outcome-only"用的是同一条代码路径、
  同一预算、同一批题；
- 结论敢说"不显著"：因为配对检验的 p=0.18 就摆在那里，而机制证据
  （`turn_adv_nonzero` 59%、hard +4.3pp）比仅报一个 pp 更有说服力。
