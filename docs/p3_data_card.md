# P3 数据卡：SFT 冷启动与 RL prompt 集

> 自动生成于 2026-11-03T16:09:38，由 `scripts/finalize_p3.py` 产出。

## 1. 内容

| 文件 | 条数 | 用途 |
|---|---|---|
| `data/processed/sft_final.jsonl` | 2574 | SFT 冷启动（chat 多轮） |
| ├ train | 2369 | 训练 |
| └ val | 205 | 验证 |
| `data/processed/rl_prompts.json` | 1500 | RL 采样题（**无答案**） |
| `data/synthetic/episodes.jsonl` | 原始轨迹 | 审计用 |

- 覆盖数据库：140 个（train split 的库，与 dev 的 20 个库不重叠）
- 难度分布：{'medium': 1249, 'easy': 719, 'extra': 253, 'hard': 353}
- 轮数分布：{1: 2412, 2: 121, 3: 41}，多轮占比 **6%**
- 估算规模：133,160 tokens

## 2. 生成方式

1. **种子**：Spider 1.0 train 集，剔除 3 条 gold SQL 本身无法执行的脏数据
   （2 条 `INTERSECT` 前误用 `ORDER BY`，1 条库中缺表）；
2. **采样**：强模型（DeepSeek-chat）在「schema + 问题」上多轮生成，温度 0.7；
3. **反馈**：每轮把 SQLite 执行结果或结构化报错回灌，**不含 gold 答案**；
4. **停止**：判对即停（最多 3 轮）；
5. **过滤**：只保留最终判对的轨迹；保留中间失败轮，让模型学到"改错"。

## 3. 已知偏差（必须写清楚）

1. **蒸馏偏差**：轨迹由强模型生成，SFT 后的 4B 模型最多逼近该模型水平；
2. **判对口径**：用单库执行等价判定，比官方 test-suite 宽松，可能混入少量
   "结果碰巧相同"的错误 SQL；
3. **难度标签是自造的**：`envs/difficulty.py` 按 SQL 结构分档，不是 Spider 官方标注；
4. **多轮样本只覆盖成功的修正**：255 条跑满 3 轮仍错的轨迹没有进 SFT
   （无成功答案），它们是 RL 阶段的主要学习对象。

## 4. 与 RL prompt 集的隔离

`rl_prompts.json` 与 `sft_final.jsonl` **来自不同种子的互斥子集**，
避免 RL 阶段在 SFT 见过的题上过拟合。两者都不含 dev 的 20 个库。
