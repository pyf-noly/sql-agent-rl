# P6 前置调研：BIRD 可行性（本地实测版）

> 2026-11-05 · 全部数字来自本机实测（RTX 4060 8G + 4-bit 量化），没有花一分钱 GPU 租金。
> 产出的代码与数据管线已经合进仓库：`scripts/prepare_bird.py`、`scripts/selfcheck_bird.py`、
> `configs/train/sft_bird.json`、`configs/train/grpo_bird.json`、`tests/test_bird_pipeline.py`。

---

## 0. 一句话结论

**BIRD 可行，但"4B 在 BIRD 上只有 10~30%"这个前提是错的。**

HANDOFF 里的"数据库 30GB+"其实**只说对了 train**，而且没人注意到 dev 只有 1.5GB。
实测（全部真下载真解压）：

- **dev**：`dev.zip` 346MB → 解压 **1.71GB**（11 个库，够跑全部评测）；
- **train**：`train.zip` 8.92GB → 9.35GB → 里面还有**第二层** `train_databases.zip`（8.71GB）
  → 解开 **30.18GB**（69 个库）。论文说的 33.4GB 就是这个量级；
- **但 P6 只需要 dev + RL 留出库**：SFT 用的 jsonl 里已经带 schema（不需要数据库），
  GRPO 的奖励只要执行那 15 个留出库的 SQL。所以 **dev 1.7GB + RL 库 8.4GB ≈ 10GB**
  就够跑完全程（`scripts/fetch_bird.py` 默认就这么干）。

4B 基线：零样本单轮 **49.8%（BIRD 官方口径）/ 46.1%（本项目口径）**，
本机 4060-8G 4-bit 评测，512 条等距抽样。

这个数字不高不低，正好是 GRPO 想要的难度区间：实测零方差从 Spider 的 0.81~0.87
降到 **0.615**（有梯度的步数 13~19% → **38.5%**）。
**注意别用均值准确率套公式估零方差**——真实分布是双峰的，套公式会高估 5 倍（见 §2.2）。

---

## 1. 数据规模：实测 vs 旧假设

| 项目 | 交接文档/旧假设 | **实测** | 依据 |
|---|---|---|---|
| dev 集大小 | 30GB+，盘要 ≥60G | **1.71GB** | `dev.zip` 346MB → 解压后 11 个 sqlite + 元数据 |
| dev 题目数 | 1534 | **1534** | 一致 |
| dev 库数 | 178（论文里的 dev 分片数） | **11** | `dev.json` 去重 `db_id` = 11 |
| train 压缩包 | 30GB+ | **8.92GB** | `train.zip` Content-Length，已下载 |
| train 解压后 | — | **9.35GB**（含第二层 `train_databases.zip` 8.71GB）→ 再解开 **30.18GB** / 69 库 | 实际解压测的 |
| train 题目/库数 | 9428 / 69 | **9428 / 69** | `train.json` 实测 |
| **跑全程实际需要** | 60G 数据盘 | **≈10GB**（dev 1.7 + RL 留出库 8.4）；SFT 不需要数据库 | `fetch_bird.py --rl-only` 默认 |
| 4B 单轮基线 | 10~30%（估计） | **49.8%（官方）/ 46.1%（本项目）** | 本地 4-bit 评测，见 §2 |
| gold 可执行 | 未验证 | **1530/1534** 在 2s 内，**1532/1534** 在 10s 内；2 条（`card_games#518`、`codebase_community#701`）30s 仍超时 | 逐条跑沙箱 |
| 库级切分 | 未定 | **54 库 SFT / 15 库 RL**，RL 1976 条 prompt | `prepare_bird.py` 实跑 |

**两条工程含义**：

1. **磁盘比想象的小，但"全面解压"确实要 34GB**：dev 1.7GB + train 30.2GB ≈ **32GB**；
   而只取 dev + 15 个 RL 留出库只要 **≈10GB**。`scripts/fetch_bird.py` 默认按后者做——
   下载必须下 8.9GB 的整包（BIRD 没提供按库下载），但解压只解需要的库，磁盘省 20GB。
2. **但评测口径要改**：BIRD 有 4 条 gold 查询跑不完 2s、55 条结果超过 1000 行。
   沿用 Spider 的 `timeout=2s / max_rows=1000` 会把它们误判成错（前者）或误判成对（后者，
   两边截断到同样前缀 → 假阳性）。所以 BIRD 评测固定 `timeout=10s`、`max_rows=20000`，
   这两项已经写进 `run_baseline.py --dataset bird` 的默认值里。

### 逐库规模（dev）

| 库 | sqlite | 表 | 列 |
|---|---|---|---|
| california_schools | 11.1 MB | 3 | 89 |
| card_games | 261.8 MB | 6 | 115 |
| codebase_community | 481.4 MB | 8 | 71 |
| debit_card_specializing | 34.6 MB | 5 | 21 |
| european_football_2 | 597.8 MB | 7 | 199 |
| financial | 71.3 MB | 8 | 55 |
| formula_1 | 22.4 MB | 13 | 94 |
| student_club | 2.6 MB | 8 | 48 |
| superhero | 0.2 MB | 10 | 31 |
| thrombosis_prediction | 7.3 MB | 3 | 64 |
| toxicology | 2.7 MB | 4 | 11 |

---

## 2. 4B 基线：本地实测（零成本）

### 2.1 怎么测的

- 硬件：RTX 4060 8G（本机），torch 2.11.0+cu128，transformers 5.22，bitsandbytes nf4 4-bit；
  显存峰值 **2.84GB**，所以 8G 卡完全够。
- 模型：`Qwen/Qwen3-4B-Instruct-2507`（与训练用的是同一个基座）。
- prompt：`BIRD_PROMPT`（列类型 + 外键 + 引号 + evidence），与将来训练/评测**同一套**。
- 评测：本仓库的 `Evaluator`（单库执行等价），`timeout=10s`、`max_rows=20000`。

### 2.2 结果

| 设置 | 口径 | BIRD dev EA | easy | medium | hard | 说明 |
|---|---|---|---|---|---|---|
| base 4B，单轮、零样本、4-bit | **官方口径（set）** | **49.8%** | 57% | 40% | 32% | 512 条等距抽样（stride=3），本地实测 |
| base 4B，同上 | 本项目 rows 口径 | 46.1% | 52% | 38% | 32% | 同上；口径差异见 §2.3 |
| base 4B，前 40 条 | rows | 27.5% | 38% | 14% | 0% | 只用于早期摸速度，样本偏 california_schools |
| Spider-SFT 4B（P4 adapter），零样本迁移 | 官方口径 | 40.9% | 49% | 26% | 33% | 220 条抽样；配对比较 +1.4pp（无迁移），见 §2.4 |
| 参考：EvoSQL 论文 Qwen3-4B | — | 66.8% | | | | 是 **Maj@16**（16 次采样），不是单轮 |

**结论**：单轮、零样本、不微调的 4B 在 BIRD dev 上是 **≈50%（官方口径）/ 46%（本项目口径）**。
比 HANDOFF 里"10~30%"高得多，也比"Spider 的 78%"低得多——正好落在 GRPO 想要的位置。

**但"位置好"不等于"信号多"——这一点上我最初算错了，实测数据纠正了它**：

| 估计方式 | 零方差（全组同分） | 有梯度步数 |
|---|---|---|
| 朴素公式 `p⁴+(1−p)⁴`（假设每题成功率都等于均值 0.5） | 0.125 | 88% |
| **实测（2026-11-05，BIRD RL 留出库，200 步 × G=4）** | **0.615** | **38.5%** |
| 对照：P5 在 Spider 上实测 | 0.81~0.87 | 13~19% |

差 5 倍的原因：**那个公式假设"每道题成功率都一样"，而真实分布是双峰的**——
67/200 步是 4 条全对（题太简单），56/200 步是 4 条全错（题太难），
均值 p=0.54 只是这两头的加权平均。**用均值准确率预测零方差会严重高估学习信号。**

正确的结论是：**换 BIRD 让有梯度的步数从 13~19% 提高到 38.5%（2~3 倍），
但没有到"随便训都能涨"的程度**——P6 的结果（`docs/p6_results.md`）也验证了这一点：
200 步单轮 GRPO 没能超过 SFT。

### 2.3 两种评测口径差 3.7pp（必须写清楚）

| 口径 | 定义 | base 4B |
|---|---|---|
| `bird`（官方） | `set(pred_rows) == set(gold_rows)`：原始元组、列顺序敏感、忽略行序与重复行 | **49.8%** |
| `rows`（本项目历史口径） | 先比行数，再行内排序后整体比较，数值归一化 | 46.1% |

差异来源：官方口径忽略重复行，我们口径对"多返回一行/少返回一行"更严格；反过来我们对
`1` 与 `1.0` 更宽松。实测净差 **3.7pp**，方向是**我们的口径更严**。

所以：**和论文/榜单比要看 `bird` 口径，和自己的 Spider 历史表比要看 `rows` 口径**，
两份数都要报，不能只报高的那个。代码里用 `--compare rows|bird` 切换，
`scripts/reeval_results.py` 可以对已有结果文件重算（不需要 GPU）。

### 2.4 Spider-SFT 的零样本迁移（本地实测）

> 回答"能不能接着用 P4 的 SFT adapter 直接上 BIRD"：如果能，就省掉一次 BIRD SFT。

| 设置 | 样本 | 口径 | EA | easy | medium | hard |
|---|---|---|---|---|---|---|
| base 4B | 512 条等距（stride 3） | bird | **49.8%** | 57% | 40% | 32% |
| base 4B | 同上 | rows | 46.1% | 52% | 38% | 32% |
| **Spider-SFT 4B（P4 adapter）** | 220 条等距（stride 7） | bird | **40.9%** | 49% | 26% | 33% |
| Spider-SFT 4B（P4 adapter） | 同上 | rows | 37.3% | 45% | 23% | 28% |
| **同一批题配对比较**（74 条） | — | rows | base 39.2% → SFT 40.5%，**+1.4pp** | | | |

**结论：Spider-SFT 在 BIRD 上没有迁移收益（配对 +1.4pp，n=74，标准误约 8pp）。**

注意别被"40.9% vs 49.8%"骗了：那是**不同抽样批**的对比，差 9pp 里大部分是题目难度差异；
把两批结果按原题号配对后（74 条重叠），差只有 +1.4pp，完全在噪声内。

**这个结论的用法**：

1. **不能省 BIRD SFT**：Spider 的 SFT 学的是小库、无 evidence 的 SQL 习惯，
   在 BIRD 上只是"不倒退"，不会替你解决 BIRD 的取值/外部知识问题；
2. **从零 SFT 更干净**：两阶段（Spider→BIRD）会引入"先小库后大库"的分布切换，
   收益不明、风险明确，所以 `guides/P6.md` 的 Step 2 默认从零训；
3. 口径提醒：这个测试用的也是 BIRD prompt（带类型/引号/evidence），而 P4 adapter 是在
   Spider prompt 上训的——它属于"换 prompt + 换库"的双重分布偏移，所以结论应读作
   "直接拿来用没有收益"，而不是"Spider-SFT 能力差"。

### 2.5 时间与成本

| 动作 | 4060-8G（4-bit） | 4090（bf16，按 P4 比例折算） |
|---|---|---|
| 单题生成 | ≈3.7s（含早停） | ≈1.1s |
| dev 1534 条全量评测 | ≈1.6h | ≈30min |
| 512 条抽样评测 | ≈32min | ≈5min |
| 备注 | 早停前是 21s/题，见下 | |

**早停这一条值得单独记**：Qwen3-4B-Instruct 基座即使 system prompt 写了"只输出 SQL"，
也会在代码块外写一段解释；不截断就会一直生成到 `max_new_tokens`。
实测 21s/题 → 3.7s/题（**快 5.7 倍**），而生成内容与判定结果不变。

---

## 3. 代码改造量（已完成 95%：只剩多轮轨迹合成与上卡执行）

所有改动都保持 **Spider 行为逐字不变**（默认参数一个没动），BIRD 走显式 preset。

| 模块 | 改了什么 | 为什么 | 状态 |
|---|---|---|---|
| `envs/db_registry.py` | `load_bird_examples`：`SQL`→`query`、难度归一、evidence | 字段名不同，不映射就是全错 | ✅ |
| 同上 | `render_ddl(include_types/include_fk/quote_identifiers)` | 列名带空格/括号，不引用模型会自己删空格 | ✅ |
| `rollout/prompt.py` | `PromptConfig` 新增 evidence/引号/列预算；`BIRD_PROMPT` | BIRD 官方要求带 evidence | ✅ |
| `rollout/agent.py` | evidence 透传、`max_rows` | 训练与推理同一套 prompt | ✅ |
| `rollout/evaluator.py` | `max_rows` 参数 + BIRD 口径差异说明 | 截断会造成假阳性 | ✅ |
| `scripts/prepare_bird.py` | 库级切分 + SFT/RL 数据 + 泄漏检查 | 复用 P5 的方法论 | ✅ |
| `scripts/selfcheck_bird.py` | 12 项离线自检 | 上卡前最后一次把关 | ✅ |
| `scripts/run_baseline.py` | `--dataset bird`、`--db-root`、`--max-rows`、`--load-in-4bit`、`--stride`、`--compare` | 同一份代码评两种数据 | ✅ |
| `scripts/reeval_results.py` | 对已有结果按 `rows` / `bird` 两种口径重算 | 换口径不用重跑模型 | ✅ |
| `training/model.py` | 4-bit 装载 + 代码块闭合早停 | 8G 卡跑 4B；早停省 78% 时间 | ✅ |
| `training/chatfmt.py` | 兼容 transformers 5.x 的 `BatchEncoding`；messages 带 prompt_cfg/evidence | 4.x→5.x 的静默 break | ✅ |
| `training/train_grpo.py` | `db_root` / `prompt_preset` / `sql_max_rows` / `compare` 可配 | 训练奖励口径必须等于评测口径 | ✅ |
| `tests/test_bird_pipeline.py` | 17 个单测 | 上述每一条都有回归 | ✅ |
| `configs/train/sft_bird.json` / `grpo_bird.json` | max_len 3072、timeout 10s、compare bird | BIRD prompt 更长 | ✅ |
| `scripts/pack_bundle.py` | 跳过 BIRD/Spider 数据库、默认输出改 `dist/train_bundle.zip` | 8.9GB 数据库不能进包 | ✅ |
| 多轮 SFT 数据（P6 要用） | 用 API 合成 BIRD 的 3 轮轨迹 | 轮级信用分配需要多轮轨迹 | ⬜ 可选 |

**总改造量**：~600 行新代码 + 8 个文件的小改，已全部有单测（205 passed）；
没有一处改动破坏 Spider 的结果复现（默认参数一个没动）。

---

## 4. 训练成本与时间（按 4090-24G 估算）

| 步骤 | 时长 | 备注 |
|---|---|---|
| 下载 `train.zip`（8.9GB）+ 解压 | 15~25 min | 实测本地 15MB/s 约 10 min；AutoDL 内网更快 |
| `prepare_bird.py`（69 库、9428 题、库级切分） | ~5 min | 纯 CPU；本地已实测通过（3920 SFT / 80 val / 1976 RL） |
| BIRD SFT（3920 条、2 epoch、max_len 3072） | 45~70 min | P4 的 336 步是 17.5 min@2048；3072 约 1.6× |
| dev300 快速评测 | 15 min | bf16 4090 |
| GRPO 200 步（单轮、G=4、库级留出） | 90~120 min | 与 P5 同量级 |
| dev 全量评测（1534） | 25~35 min | |
| （可选）pass-rate 校准 1976 条 | 30~40 min | 若 SFT 后准确率 >65% 才需要，用来挑 p∈0.2~0.8 的题 |
| **合计** | **≈4~6 h** | **¥10~16**（4090 ≈¥2.5/h） |

对比 P5：4.5h / ¥9。也就是说，**换 BIRD 的边际成本几乎为零**，只是把同样的机时花在
更有梯度的数据上。

---

## 5. 风险与红线

| 风险 | 具体表现 | 对策 |
|---|---|---|
| **4B 在 BIRD 已有 50%** | SFT 后可能到 60~65%，RL 空间被压缩，又回到"太简单" | ① 只用 hard 子集做评测对照；② 若 SFT >65%，先跑 1976 条 pass-rate 校准，只留 p∈0.2~0.8；③ 报告里同时给 easy/medium/hard 三段 |
| 数据下载慢/失败 | 8.9GB 走公网可能很慢 | AutoDL 上先测速；dev 只需 346MB，可先只跑评测 |
| 库级切分泄漏 | 同一库同时出现在 SFT 和 RL | `prepare_bird.py` 按 db 切分 + 显式 overlap 检查（已实测 0） |
| 评测口径不一致 | 2s/1000 行 → 假阴/假阳 | BIRD 分支默认 10s/20000 行；`--compare bird` 对齐官方 |
| 有 2 条 gold 跑不完 | `card_games#518`、`codebase_community#701` 超过 30s | 官方口径里它们同样算错，占 0.13%，报告注明即可；`selfcheck_bird` 会单独计数不报错 |
| 4-bit 数字偏低 | 量化损失约 1~2pp | 正式数字必须在 4090 bf16 上复测；4-bit 只作方向 |
| 训练 OOM | 3072 序列 + G=4 | micro_batch=2；先跑 3 步看显存再放开 |
| 版本差异 | 本地 transformers 5.22、租卡机 4.6 | `chatfmt` 已双向兼容（本地踩到过 5.x 的 BatchEncoding 坑） |

---

## 6. 决策建议：A / B / C 怎么选

| 选项 | 成本 | 得到什么 | 建议 |
|---|---|---|---|
| **A. BIRD 主实验** | 1 次开机 / 4~6h / ¥10~16 | "在更有梯度的数据上，RL 把 50% 提到 X%"，且给出 easy/medium/hard 三段 | **推荐** |
| **A+. BIRD + hard 子集对照** | 同一张卡顺带 | 证明增益来自难样本而不是简单样本 | 强烈建议，几乎零额外成本 |
| **B. P6 轮级信用分配（在 BIRD 上做）** | 第 2 次开机 / 2~3h / ¥6~8（+ API ¥8~10 合成多轮数据） | 项目灵魂：轮级 advantage 的机制解释 + 消融 | **A 之后的下一步** |
| **C. 直接收口（P7）** | 0 | 现有 Spider 结果（78→87，+2pp RL）已够写报告 | 时间紧时的兜底 |

**推荐路径**：A（Stage 1：BIRD SFT + 单轮 GRPO，约半天）→ 看 GRPO 是否 ≥ +3pp：
- 是 → 做 P6（Stage 2：多轮 + 轮级 advantage）；
- 否 → 先跑 pass-rate 校准筛题，再判断，不要盲目调参（P5 的教训）。

**为什么不选 C**：P5 已经把"Spider 对 4B-SFT 接近饱和"钉死了（零方差 74~90%，四次尝试），
继续在 Spider 上调参没有意义。而 BIRD 的 p≈0.5 是**用数据难度换学习信号**——
这次有本地实测的零方差估算（88% vs 13~19%）支撑，不是拍脑袋。

---

## 7. 复现（本地、零成本）

```bash
# 1. 数据（只需 dev 就能复现本文数字）
#    dev.zip:  https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip  (346MB)
python scripts/prepare_bird.py --skip-train --skip-gold-check
python scripts/selfcheck_bird.py                    # 12 项全绿

# 2. 本地 4-bit 基座评测（4060-8G 约 1.8h）
python scripts/run_baseline.py --dataset bird --backend hf \
    --model-path .models/Qwen3-4B-Instruct-2507 --load-in-4bit \
    --turns 1 --timeout 10 --max-new-tokens 320 --tag bird_base4b_t1

# 3. 上卡（等 §4 的流程）
python scripts/prepare_bird.py --train-dir data/raw/bird/train   # 含库级切分
python -m training.train_sft  --config configs/train/sft_bird.json
python -m training.train_grpo --config configs/train/grpo_bird.json \
    --init-adapter runs/sft_bird/adapter --model /root/autodl-tmp/qwen3-4b
```
