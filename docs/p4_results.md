# P4 结果记录：第一次真训练

> 环境：AutoDL 单卡 RTX 4090 24G（PyTorch 2.5.1 + CUDA 12.4）· 01:14 GPU 时长 · 约 ¥6
> 模型：Qwen3-4B-Instruct-2507 + LoRA(r=16, α=32)，可训练 33.03M（0.81%）

## 1. SFT（E1 冷启动）

| 项 | 值 |
|---|---|
| 数据 | train 2369 条（多轮过采样后 2673）/ val 205 条 |
| 步数 / 耗时 | 336 步 / 1048s（3.1s/步） |
| val_loss | **0.0728**（train 0.037，未过拟合） |
| 显存峰值 | 12.18G |
| 序列特点 | 平均 336 token，其中 **88.7% 是 schema 等上下文**，助手 token 只占 11.3% |

## 2. 评测（dev300，20 个训练未见过的库）

| 设置 | EA | easy | medium | hard | extra | 执行失败 |
|---|---|---|---|---|---|---|
| **4B-SFT 单轮** | **78.00%** (234/300) | 88% | 84% | 66% | 57% | 19 |
| **4B-SFT 三轮**（判对才停） | **87.00%** (261/300) | 92% | 89% | 83% | 78% | 15 |
| 增益 | **+9.00pp** | +4 | +5 | **+17** | **+21** | −4 |

对照 DeepSeek-chat（P2）：

| | 单轮 | 三轮 | 增益 |
|---|---|---|---|
| DeepSeek-chat | 84.33% | 89.00% | +4.67pp |
| **Qwen3-4B + LoRA** | 78.00% | 87.00% | **+9.00pp** |

**解读**：
- 4B 单轮比 DeepSeek 低 6.3pp，差距集中在 hard（−8）与 extra（−27）；
- 4B 的多轮增益 **比 DeepSeek 还大**（+9.0 vs +4.67），因为它的错误更多是"聚合口径/JOIN 条件"这类
  **执行反馈能定位**的问题；DeepSeek 剩下的错多是语义理解，反馈救不了。
- 三轮 87.0% 已逼近 DeepSeek 的 89.0%。

## 3. GRPO 冒烟（20 步，outcome-only）

| 指标 | 值 | 判定 |
|---|---|---|
| `mask_mismatch` | 0 | ✅ 梯度可信 |
| `ratio_mean` | 1.0（全程） | ✅ 单步更新，说明老策略 logp 算法正确 |
| 显存峰值 | 9.85G | ✅ |
| 平均轮数 | 1.25 | 87% 的题第 1 轮就答对 |
| `reward_mean` | 0.95 | 训练集（SFT 见过的库）上几乎全对 |
| **`zero_var_ratio`** | **0.90** | ⚠️ **90% 的组没有梯度信号** |

### 这是 P4 最重要的发现

`reward_mean = 0.95`，G=4 时"四条全对"的概率 ≈ `0.95⁴ ≈ 0.81`，加上"全错"的组，实测 0.90。
优势恒为 0 → 这批采样对参数更新零贡献。

**结论：outcome-only GRPO 在 Spider-train 这个难度、这个模型上，几乎学不到东西。**
这不是实现 bug（mask、ratio、显存全部正常），是"任务偏简单 + 训练集被 SFT 覆盖"的必然结果。

## 4. 对 P5 的直接影响

P5 不能直接开训 300 步，必须先造"有区分度"的数据：

1. **pass-rate 校准**：对 hard/extra 的 396 条 prompt 各采 G=4~8，统计每题通过率；
2. **只保留 `0.2 < p < 0.8` 的题**（太简单没信号、太难全错也没信号）；
3. **G 提到 8**，把全对概率从 0.81 压到 0.66；
4. 若筛完不足 80 条，则改用**分层奖励**（首轮对 + 修复成功分开给分），直接进入 P6 的轮级设计。

## 5. 工程上踩过的坑（都已修进代码）

| 坑 | 症状 | 修法 |
|---|---|---|
| SFT 数据缺 system/schema | 训练分布与推理不一致 | `scripts/build_p4_data.py` 重建 |
| 生成时 `use_cache=False` | rollout 慢 5 倍、显存暴涨 | `generate_text` 显式 `use_cache=True` |
| 更新时停在 `eval()` | 梯度检查点不生效 → OOM | `model.train()`；检查点必须在 peft 包装**前**开在基座上 |
| 模型对象写进 JSON | 300 条评测结果全丢 | `default=str` + 每 25 条落盘 |
| `hf_transfer` + 代理 | 下载直接失败 | 改走 ModelScope（7MB/s，稳定） |

## 6. 复现命令

```bash
python scripts/build_p4_data.py
python -m training.train_sft --config configs/train/sft_qwen3_4b.json --model /root/autodl-tmp/qwen3-4b
python scripts/run_baseline.py --backend hf --turns 1 --adapter runs/sft_qwen3_4b/adapter \
    --model-path /root/autodl-tmp/qwen3-4b --tag sft_t1
python scripts/run_baseline.py --backend hf --turns 3 --until-correct --adapter runs/sft_qwen3_4b/adapter \
    --model-path /root/autodl-tmp/qwen3-4b --tag sft_t3
python -m training.train_grpo --config configs/train/grpo_smoke_4b.json \
    --init-adapter runs/sft_qwen3_4b/adapter --model /root/autodl-tmp/qwen3-4b --group-size 4 --micro-batch 1
```
