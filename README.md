# sql-agent-rl

**面向 Text-to-SQL 的多轮自我修复 Agent：环境、数据、SFT / GRPO 训练与轮级信用分配。**

一个 4B 模型（Qwen3-4B-Instruct-2507 + LoRA）在只读 SQLite 沙箱里"写 SQL → 看执行反馈 → 改 SQL"，
用自研轻量 GRPO 训练；核心改动是把 **outcome-only 的整条轨迹优势拆到每一轮**，
并给出它有效 / 无效的机制证据。

一句话结论：**换到 BIRD 之后，三轮自修复的 +2.93pp 增益来自"反馈里终于有信息了"
（取值存在性检查），而不是来自训练步数；轮级优势方向对但单 seed 不显著。**

| 数据集 | 指标 | 结果 |
|---|---|---|
| Spider dev300 | 4B-SFT 单轮 → 三轮 | **78.00% → 87.00%（+9.00pp）** |
| Spider dev300 | GRPO（2 seed，单轮） | **+2.00pp**（未达显著） |
| BIRD dev（**全量 1534**） | v3 单轮 | **50.85%**（±1.3pp；307 抽样是 51.79%，偏高 0.94pp） |
| BIRD dev（307 条同批） | 三轮自修复（v3，最终配置） | **51.79% → 57.00%**（多轮 +5.21pp，p<0.001；较上一代数据 +2.93pp，p=0.27） |
| BIRD dev（307 条同批） | 轮级优势 vs outcome-only（单轮） | **52.77% vs 51.14%（+1.63pp；p=0.18）** |

> 数字都在 `results/*.json`，可离线重算（`scripts/reeval_results.py`），本文不放"手抄"的数。

---

## 1. 它解决什么问题

多轮 Agent 的 RL 有个结构性缺陷：**同一个奖励分不出"第一轮就对了"和"第三轮才改对"**。
GRPO 把一条轨迹的所有 token 赋同一个优势（outcome-only），于是"过程"没有被学习的入口；
更糟的是，当一组 4 条轨迹**最终都对**时，优势恒为 0——而这组里"一轮解出"和"三轮挣扎"
的差别恰恰是最该学的。

本项目做了两件事：

1. **把优势拆到轮**：`A[i,t] = A_ep + β·γ^(t-1)·normalize(首次做对的轮次)`，
   β=0 严格退化为 outcome-only，所以消融免费；
2. **把反馈的信息量做上去**：执行反馈只能发现"跑不起来"的错误——BIRD 首轮错题里
   **79%~90% 是"跑得对但结果错"**（307 条评测 117/149，抽样 26/29），执行反馈完全看不见；
   加上**取值存在性检查**（`envs/value_check.py`）后，其中"值拼错"这一类
   也变成可检测信号，且不泄漏答案。

---

## 2. 方法

```
                    ┌──────────────────────────────────────────────┐
  question + schema │  SqlAgent（rollout/agent.py）                 │
  ─────────────────▶│   模型输出 ```sql 块                          │
                    │   → SqlSandbox 只读执行（2s 软超时）           │
                    │   → 错误分类（9 类，envs/error_taxonomy.py）    │
                    │   → 反馈文本（envs/feedback.py，唯一出口）      │
                    │   → 值检查（可选，envs/value_check.py）        │
                    │   → 回灌，最多 3 轮                           │
                    └───────────────┬──────────────────────────────┘
                                    │ Episode（逐轮 SQL / 反馈 / 是否首次做对）
                                    ▼
              SFT（LoRA，训练/推理共用同一套 messages）
                                    │
                                    ▼
              GRPO（training/，约 1400 行，单卡 4090 可跑）
                loss = -min(ratio·A, clip(ratio)·A)，A 可为 [B] 或 [B, L]
```

### 2.1 环境（P1）

- **只读**：SQLite `file:...?mode=ro`——物理层防线，不依赖我们的解析代码；
- **单语句 + 白名单**：`complete_statement` + `sqlparse.split` 双层校验，只放行 `SELECT/WITH`；
- **软超时**：`set_progress_handler` 每 2000 条 VM 指令检查一次 deadline，超时返回结构化
  `error_class="timeout"`，不杀进程；
- **结果截断**：`fetchmany(max_rows+1)` 检测截断，避免"截断后前缀相同"被判成等价；
- **反馈**：9 类错误各有模板，schema 类错误附真实列名，**绝不包含 gold SQL 或正确答案**
  （有专门的防泄漏测试）；
- **值检查**：执行成功后额外查 `WHERE` 里的字符串字面量是否真的存在于库中，
  不存在就给最近的真实取值。这是"环境看得见、模型看不见"的合法信号。

### 2.2 数据（P3 / P6）

| 数据 | 规模 | 用途 |
|---|---|---|
| Spider dev300 分层抽样 | 300 条（seed 42） | 主表评测，20 个训练未见过的库 |
| Spider SFT 种子 | 2574 条（140 库） | SFT 冷启动 |
| BIRD dev | 1534 条 / 11 库 | 评测（BIRD 官方 EX 口径） |
| BIRD train | 9428 条 / 69 库 | 库级切分：54 库 SFT / 15 库 RL（零重叠） |
| BIRD P6 多轮 SFT | **3918 条**（单轮 1990 + 多轮修复 1928） | 多轮修复行为 |

多轮修复轨迹是**合成**的，不是"强模型改错"（那条路实测只 10% 成功率）：

| 方案 | 做法 | 实测成功率 |
|---|---|---|
| ① 强模型直接做 3 轮 | DeepSeek 跑"判对才停" | 首轮 3/3 全对，没有修复轨迹 |
| ② 强模型改 4B 的真实错误 | 4B 先试 → 反馈 → DeepSeek 改 | **10%**（4B 首轮错题 79%~90% 是"执行成功但结果错"，反馈里没有可用的纠错信息） |
| ③ **构造可检测错误**（采用） | 把 gold 的列名换成本库另一张表的真实列 → 真报 `no such column` → 目标仍是 gold | **94%**（1133/1200） |
| ③' 值修复（采用） | 把 gold 里的真实取值改坏（`Yes`→`Yse`）→ 值检查给出最近真实值 | **99%**（788/795） |

关键细节：修复样本带 `meta.mask_first_turn=True`，SFT **只对最后一轮算 loss**。
不 mask 会教模型"先写错等反馈"，实测单轮掉 4.6pp。

另一条被实验否掉的路线：直接拿 Spider-SFT 的 adapter 迁到 BIRD，同题配对只 **+1.4pp（n=74）**——
跨数据集没有迁移，所以 BIRD 从零 SFT，而不是在 Spider 模型上继续训。

### 2.3 训练（P4~P6）

- **SFT**：LoRA(r=16, α=32)，2 epoch，max_len 2560，梯度检查点；
- **GRPO**：自研（`training/`），G=4、lr 5e-6、KL=0、T=1.0、1 个 inner epoch（ratio≡1）；
- **为什么不用 verl / vLLM**：P6 要改 advantage 的计算方式，框架越重改动越痛；
  单卡 300 步也用不着异步 rollout。代价是自己维护 1400 行训练栈（230 个单测兜底）。

**轮级优势实现**（`training/grpo.py:84-159`）：

```python
y[i, t] = 1                     # 当且仅当第 i 条轨迹"第一次做对"发生在第 t 轮
A_turn[i, t] = (y[:,t] - mean_t) / std_t * gamma**(t-1)
A[i, t] = A_ep[i] + beta * A_turn[i, t]      # beta=0 → 严格退化为 outcome-only
loss = -min(ratio*A, clip(ratio)*A)           # token 级，按轮次铺开
```

- 组内该轮全同（std≈0）→ 置 0，坚持"没方差就不给梯度"；
- `advantages` 支持 `[B]` 和 `[B, L]` 两种形状，后者要求 `adv[:, 1:]` 与平移后的 logp 对齐
  （这里错过一次：`578 vs 579`）；
- 只在"全对组"里制造非零信号——这正是 outcome-only 浪费掉的方差；"全错组"轮级也救不了。

---

## 3. 结果

### 3.1 Spider dev300（20 个未见库）

| 设置 | 单轮 EA | 三轮 EA | 说明 |
|---|---|---|---|
| DeepSeek-chat（API） | 84.33% | 89.00% | 对照，硬基线 |
| Qwen3-4B + SFT（140 库） | **78.00%** | **87.00%** | 多轮增益 +9.00pp（hard +17 / extra +21） |
| SFT（110 库，30 个库留给 RL） | 75.33% | 83.67% | 数据变少的代价 −2.67pp |
| + GRPO（seed 7 / seed 8） | 78.33% / 76.33% | 85.33% / — | **均值 +2.00pp**，n=300、SE≈2.4pp，未显著 |

Spider 上零方差 0.74~0.90：**任务对 4B-SFT 已接近饱和**，这是换数据集（P6）的直接原因。
P5 还试过 pass-rate 校准（只留 `0.2 < p < 0.8`）：校准 40 条里 p 均值 **0.90**、
35 条全对、3 条全错，只有 **2 条（5%）**落在有信号区间——筛完训练集只剩 2 条，
于是放弃校准、改用"库级留出 + 单轮"的口径。

### 3.2 BIRD dev（307 条等距抽样，同一批样本，BIRD 官方 EX 口径）

| 设置 | 单轮 | 三轮（判对才停） | 多轮才救回 | 三轮执行失败 |
|---|---|---|---|---|
| base 4B（512 条 4-bit） | 49.8% | — | — | 45 |
| 单轮 SFT（490 步） | 50.16% | 53.09% | 9（2.9%） | 19 |
| + GRPO outcome-only（200 步） | 50.16% | — | — | 23 |
| 多轮 SFT v1（1133 条 schema 修复） | 51.47% | 54.07% | 8（2.6%） | 19 |
| + GRPO outcome-only（120 步） | 51.14% | — | — | 34 |
| **+ GRPO 轮级优势（120 步）** | **52.77%** | **56.03%** | 10 | 24 |
| 多轮 SFT v2（+795 条取值修复，未 mask） | 46.91% | 55.37% | 26（8.5%） | 24 |
| **多轮 SFT v3（同数据 + mask 第一轮）** | **51.79%** | **57.00%** | **16（5.2%）** | **13** |
| **v3 全量 t1（1534 条）** | **50.85%** | — | — | 151 |

v3 是最终配置：三轮 **57.00%**（比 v1 **+2.93pp**），同时单轮回到 51.79%（比 v1 还高 0.3pp）。
全量 1534 条的单轮是 **50.85%**（SE 1.28pp，95% CI [48.35, 53.35]），
说明 307 条等距抽样偏乐观了 **0.94pp**；分难度：easy 58.8% / medium 39.2% / hard 37.2%，
执行失败 151 条（9.8%）。**307 子集与全量在 307 条上逐条一致**，评测是确定性的。

### 3.3 配对显著性（同一批 307 条，McNemar 精确检验）

| 对比 | 差值 | 配对 SE | 精确 p | 95% CI |
|---|---|---|---|---|
| v3 t3 − v1 t3 | **+2.93pp** | 2.37pp | 0.272 | [−1.72, +7.58] |
| 轮级优势 − outcome-only（t1） | **+1.63pp** | 0.98pp | 0.180 | [−0.29, +3.54] |
| 轮级优势 − 多轮 SFT（t1） | +1.30pp | 1.03pp | 0.344 | [−0.72, +3.32] |
| v3 t3 − v3 t1（多轮本身的收益） | **+5.21pp** | 1.30pp | **<0.001** | [+2.66, +7.77] |

**诚实结论**：唯一显著的是"多轮比单轮好"（+5.21pp）。**"v3 比 v1 好"和"轮级优势比
outcome-only 好"都只是方向一致的正向信号，n=307 单 seed 不足以定论**——按配对检验，
+2.93pp 需要约 4 倍样本量才能到 0.05。README 里不放"显著"二字。

### 3.4 服务化指标（延迟 / 吞吐 / 成本）

单卡 4090、bf16、batch=1、HF `generate`（没有 vLLM / 连续批处理）：

| 结果 | n | 平均轮数 | p50 | p95 | p99 | 吞吐（题/分） | 每千题成本 |
|---|---|---|---|---|---|---|---|
| v3 单轮（全量） | 1534 | 1.00 | **2.44s** | **4.09s** | 5.60s | **23.4** | **¥1.78** |
| v3 三轮（307 抽样） | 307 | 1.92 | 3.24s | 11.33s | 14.72s | 11.7 | ¥3.55 |

- **三轮把 p95 拉高 2.8 倍**（4.09s → 11.33s），但 p50 只涨 33%——贵在尾部，不在典型值；
- **每千题成本翻倍**，换 +5.2pp 准确率，折合"每多对一个答案 ¥0.0027"；
- 本地单卡成本（¥1.78/千题）与 DeepSeek API（≈¥1.69/千题）**同一量级**——
  本地跑省的不是钱，是数据不出域与可迭代；
- 复现：`python scripts/latency_report.py results/baseline_bird_p6v3_t1_full.json --gpu-price 2.5`
- 单题演示：`python scripts/demo_agent.py --index 2 --turns 3 --value-check --backend hf ...`
  （详见 [`docs/serving_metrics.md`](docs/serving_metrics.md)）

**B-lite：换 vLLM 服务后（同机 4090D、同 220 条、同参数、关缓存）**

| 配置 | 吞吐（题/分） | p50 | p95 | 每千题成本 | 加速 |
|---|---|---|---|---|---|
| HF 单卡串行 | 55.9 | 0.89s | 1.97s | ¥0.75 | 1.00x |
| vLLM `workers=1` | 93.1 | 0.53s | 1.16s | ¥0.45 | 1.67x |
| vLLM `workers=4` | 318.8 | 0.60s | 1.29s | ¥0.13 | 5.70x |
| **vLLM `workers=8`** | **391.7** | 0.65s | 1.41s | **¥0.11** | **7.01x** |
| vLLM `workers=16` | 415.1 | 0.73s | 1.70s | ¥0.10 | 7.42x（8→16 只 +6%，饱和） |

一致性：vLLM 与 HF 逐题判定一致 **217/220（98.6%）**；3 条差异来自贪心解码的数值差异。
注意这是**评测吞吐**，不是训练 rollout 吞吐（B-full 要权重同步 + staleness 控制）。

**B-full 第一步：rollout 采样基准**（25 题 × G=4，3 轮，同 agent）：

| 模式 | 样本/秒 | 单步 rollout | 相对 HF |
|---|---|---|---|
| HF 串行（训练现状） | 0.516 | 5.30s | 1.00x |
| HF + 4 线程（对照） | 0.126 | 12.92s | 0.24x（更慢） |
| vLLM 异步 G=4 | 2.145 | 0.91s | 4.16x |
| **vLLM 异步 K=4（16 请求在飞）** | **3.409** | **0.77s** | **6.61x** |

结论：**HF 多线程是负优化（GPU 18%），必须服务化**；异步 rollout 让单步 rollout 从 5.3s 降到 0.77s。
还没做权重同步与 staleness（B-full 第二步）。

**B-full 第二步：异步训练 A/B**（同 SFT 起点、30 步、max_len 1536）：

| 实验 | 每步 mean | p50 | rollout | update | sync | EA（154 条） |
|---|---|---|---|---|---|---|
| 同步 HF | 22.35s | 17.3s | 20.77s | 1.55s | 0 | 57.79% |
| **异步 vLLM + LoRA 热同步** | **11.65s** | **6.9s** | 9.62s | 1.65s | 0.33s | 57.14% |

**1.92x 加速（p50 2.5x），EA 在噪声内不掉，`mask_mismatch=0`、`ratio_mean` 全程 1.0**
（每步同步权重，严格 on-policy）。LoRA 热同步只花 0.33s/步。

---

## 4. 快速开始

### 4.0 环境

```bash
git clone <repo> && cd sql-agent-rl
python -m venv .venv && . .venv/Scripts/activate     # Windows；Linux 用 .venv/bin/activate
pip install -r requirements-train.txt                # torch/transformers/peft + 评测运行时
cp .env.example .env                                 # 填 OPENAI_BASE_URL / API_KEY / MODEL
python check_env.py                                  # 7 项自检
```

- **只评测**：`torch` 可以装 CPU 版，甚至不需要（`--backend api`）；
- **训练**：Linux + NVIDIA，单卡 24G 可跑 4B LoRA（P4~P6 全程 4090-24G）；
- **本地 8G 显卡**：4-bit 装 4B 只要 2.84GB 显存（`--load-in-4bit`）。

### 4.1 单元测试（不需要 GPU，也不花钱）

```bash
python -m pytest tests -q        # 230 passed
```

> Windows 注意：本仓库的测试会在临时目录建 `0700` 的库，**在受限沙箱里会失败**；
> 用普通终端跑。

### 4.2 评测

```bash
# Spider dev300：单轮 / 三轮（判对才停）
python scripts/run_baseline.py --turns 1 --tag dev300_t1
python scripts/run_baseline.py --turns 3 --until-correct --tag dev300_t3

# BIRD dev：本地 4-bit 4B，等距抽样（dev.json 按库排序，必须 stride）
python scripts/run_baseline.py --dataset bird --backend hf \
    --model-path ../.models/Qwen3-4B-Instruct-2507 --load-in-4bit \
    --turns 1 --stride 3 --timeout 10 --tag bird_base4b_s3

# 换口径重算已有结果（不用重新生成，不用 GPU）
python scripts/reeval_results.py results/baseline_bird_*.json --compare both
```

### 4.3 训练（Linux + GPU）

```bash
# 环境自检（上卡第一条命令）
python scripts/check_p6_env.py --model /root/autodl-tmp/qwen3-4b

# SFT：BIRD 多轮修复数据（3918 条，v3 配置 = mask 第一轮）
python -m training.train_sft --config configs/train/sft_bird_p6.json --out runs/sft_bird_p6v3

# GRPO：轮级优势（beta=0 就是 outcome-only 消融）
python -m training.train_grpo --config configs/train/grpo_bird_turn.json \
    --init-adapter runs/sft_bird_p6v3/adapter --model /root/autodl-tmp/qwen3-4b --steps 120
```

### 4.4 数据

```bash
python scripts/fetch_bird.py --out data/raw/bird            # dev + 15 个 RL 库（~10G）
python scripts/prepare_bird.py --skip-gold-check --holdout-dbs 15 --sft-limit 4000
python scripts/make_repair_synthetic.py --limit 1200        # schema 修复轨迹（94%）
python scripts/make_repair_value.py --limit 1500            # 取值修复轨迹（99%）
python scripts/merge_p6_sft.py                              # → data/bird_p6/
```

---

## 5. 两种评测口径（用它之前必读）

同一个模型输出，换判定口径分数会差 1~4pp：

| 口径 | 规则 | 用在哪 |
|---|---|---|
| `rows` | 行数一致 + 行内/行间排序后比较 + 数值归一化（`1 == 1.0`） | Spider（P2~P5） |
| `bird` | BIRD 官方 `set(pred) == set(gold)`：原始元组、列序敏感、忽略行序与重复行 | BIRD（P6） |

两者都不等价于官方 test-suite。**`rows` 偏宽松**（列顺序不敏感、数值归一化），
理论上会把"结果碰巧相同"的错误 SQL 判对；BIRD 的 `bird` 口径更接近论文数字。
做跨论文对比时只报 `bird`，和本项目历史表对比时报 `rows`，并写清楚。

---

## 6. 仓库结构

```
sql-agent-rl/
├── envs/                   # P1：环境（不 import 任何训练框架）
│   ├── sql_sandbox.py      #   只读 + 单语句 + 软超时 + 结果截断
│   ├── error_taxonomy.py   #   报错 → 9 类（纯函数）
│   ├── feedback.py         #   类别 → 回灌文本（唯一出口 + 消融开关）
│   ├── value_check.py      #   取值存在性检查（P6）
│   └── db_registry.py      #   schema 读取（DDL 渲染，带缓存）
├── rollout/                # P1~P2：交互与评测
│   ├── agent.py            #   单轮/多轮共用一条代码路径
│   ├── evaluator.py        #   执行等价（rows / bird 两口径）+ 指标
│   ├── prompt.py           #   Spider/BIRD 两套 preset（默认值即历史配置）
│   └── parse_sql.py        #   从输出里抽 ```sql
├── training/               # P4~P6：约 1400 行自研训练栈
│   ├── grpo.py             #   组内优势 / 轮级优势 / 裁剪目标（numpy 可单测）
│   ├── chatfmt.py          #   训练与推理逐字一致的序列构造
│   ├── train_sft.py        #   LoRA SFT（支持 mask 第一轮）
│   └── train_grpo.py       #   rollout → 优势 → 更新（单卡）
├── scripts/                # 数据、评测、自检、打包（30 个脚本）
├── configs/train/          # 12 份训练配置（每份对应一次真实实验）
├── tests/                  # 230 个单测
├── docs/                   # 阶段结果与决策记录（见下）
└── results/                # 全部评测 JSON（含逐条 Episode）
```

---

## 7. 实验记录（按阶段）

| 文档 | 内容 |
|---|---|
| `docs/p2_decision.md` | 为什么保留 Spider 不换 BIRD、评测器可信度 |
| `docs/p3_summary.md` | 数据合成与失败分析（"82% 的失败轨迹里模型改了 SQL 但没改对"） |
| `docs/p4_results.md` | 第一次真训练：SFT 78%/87%，GRPO 零方差 0.90 |
| `docs/p5_results.md` | 四次尝试把零方差压到 0.74，最终 +2.00pp（不显著） |
| `docs/p6_bird_feasibility.md` | 换 BIRD 前的零成本实测（数据规模、4B 基线 49.8%、口径差异） |
| `docs/p6_results.md` | P6 全部结果：v1/v2/v3、轮级优势 vs outcome-only、值检查 |
| `docs/p1_selftest_answers.md` | P1 遗留的 14 个设计问题（环境为什么可信） |
| `docs/p7_statistics.md` | 配对检验（McNemar）、置信区间与功效分析 |
| `docs/resume_bullets.md` | 简历三行 + 每行的证据 + 不能写的说法 |
| `docs/interview_qa.md` | 面试 10 问的口径（含"为什么 4B 能出结论""给 8 卡一个月做什么"） |
| `docs/project_walkthrough.md` | **小白版全解**：从"什么是 Text-to-SQL"讲到每个数字怎么读 |
| `docs/serving_metrics.md` | 服务化指标：延迟分位数、吞吐、每千题成本、单题演示 |
| `docs/p7_gap_analysis.md` | 三个历史会话的考古：哪些"量化收益"没写进简历、哪些操作细节不该写 |
| `docs/technical_report.md` | 完整技术报告（P0~P6 的叙事、机制分析、局限） |

---

## 8. 已知局限

1. **样本量**：BIRD 的**对照结果**是 n=307、单 seed（单轮主数字已补全量 1534）；
   +2.93pp 的配对 p=0.27，按当前效应量需要约 770 条（80% 功效约 1577 条）才能定论。
   **不声称显著**；
2. **口径**：`rows` 口径偏宽松（非官方 test-suite）；BIRD 用的是官方 `set` 口径的再实现，
   与官方脚本在超时/类型处理上仍有细微差别；
3. **4B 的天花板**：BIRD 上 4B 零样本 49.8%，剩下的错题一半以上是"语义理解"
   （join 错字段、聚合口径），执行反馈本身携带不了这类信息——这不是 RL 能补的；
4. **多轮**：v3 的"多轮才救回"16/307 = 5.2%，说明模型仍不擅长用反馈；
   真正提升它需要更长的反馈（跨查询自洽检查），本项目没做；
5. **未做**：官方 test-suite 复核、多 seed、Spider test200 复评。
   （BIRD 全量 1534 条单轮已补测：50.85% ± 1.3pp。）

---

## 9. 成本

| 项 | 金额 | 说明 |
|---|---|---|
| API（DeepSeek，数据合成 + 基线） | ≈¥8.7 | P2 ¥0.36 + P3 ¥8.3 |
| GPU 机时（AutoDL 4090-24G，≈¥2.5/h） | ≈¥50 | P4~P7 约 20 小时（含全量评测 1.1h）；P6 后半段机时记录有重叠，取估算 |
| 本地评测（4060-8G 4-bit） | ¥0 | P6 的可行性调研与全部方向性评测 |
| **合计** | **≈¥58** | 单卡、4 个自然日、无 verl、无 vLLM |

---

## 10. 复现提示（踩过的坑，都是静默的）

| 坑 | 现象 | 正确做法 |
|---|---|---|
| KL 的参考策略是基座不是 SFT | `kl` 1.7~2.6，每步把模型拉回未 SFT | **`kl_beta=0`** |
| `use_cache=False` 被 rollout 继承 | 生成慢 5 倍、OOM | `generate_text` 显式 `use_cache=True` |
| 更新时停在 `eval()` | 梯度检查点不生效 → OOM | `model.train()`，检查点在 peft 包装**前**开 |
| 同卡残留进程 | 新任务 OOM 或慢 4 倍 | 开训前 `nvidia-smi` 必须 0 MiB |
| 多轮数据把"故意写错的第一轮"算进 loss | 单轮掉 4.6pp | `meta.mask_first_turn=True` |
| 逐 token 优势形状 | `size of tensor a (578) must match b (579)` | `adv[:, 1:]` 与平移后的 logp 对齐 |
| BIRD 列名带空格 | `Free Meal Count (K-12)` 写成裸标识符 → 语法错 | `BIRD_PROMPT` 打开 `quote_identifiers` |
| BIRD 慢查询 / 大结果 | 4 条 gold 超 2s、55 条结果 >1000 行 | 评测固定 `--timeout 10 --max-rows 20000` |
| `nohup` 跑训练"日志卡住" | stdout 是块缓冲，8KB 才落盘 | 看 `metrics.jsonl`（每步关文件，实时） |

---

## 11. 许可与致谢

代码 MIT。数据集遵循各自许可：Spider 1.0（CC BY-SA 4.0）、BIRD（CC BY-SA 4.0）。
基座模型 Qwen3-4B-Instruct-2507 遵循 Apache-2.0。
本项目不包含任何 Spider/BIRD 的数据库文件，数据需按 §4.4 自行下载。
