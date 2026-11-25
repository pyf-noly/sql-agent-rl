# P4 · Step 1 租卡：从注册到连上机器（零基础版）

> 这一篇只干一件事：**让你在 20 分钟内拥有一台能用 GPU 的 Linux 机器**，并且不花冤枉钱。
> 后面所有训练命令都在这一篇的最后一步之后才开始。
>
> 平台以 **AutoDL** 为例（国内最容易上手、有"无卡模式"、按小时计费）。
> 界面文字会随时间变，但**逻辑不变**：选卡 → 选镜像 → 选盘 → 选计费 → 开机 → 连上。

---

## 0. 先做个决定：租什么

| 选项 | 建议 | 理由 |
|---|---|---|
| **RTX 4090D / 24G** | ✅ **首选** | ¥1.88/h 左右，24G 装得下 4B+LoRA；4090D 比 4090 好抢，性能几乎一样 |
| RTX 4090 / 24G | 可以，但常常没货 | ¥1.98/h 左右，和 4090D 用起来没差别 |
| RTX 3090 / 24G | 备选 | ¥1.2/h 更便宜，但更慢、更老；P4 就 2~3 小时，省不了几块钱 |
| A100 80G | ❌ 不要 | ¥5.5/h 起，P4 用不上；这是 P5 跑 4B 大规模时才考虑的 |
| 包日/包周 | ❌ 不要 | 你 80% 时间在本地写代码，包时会浪费 |

**预算**：P4 全程 2~3 小时 GPU（¥5~6）+ 无卡模式 1 小时（¥0.1）+ 模型下载流量 ≈ **¥8~15**。
充值 **¥30** 足够，别一次充几百。

---

## 1. 注册、实名、充值

1. 打开 <https://www.autodl.com/>，手机号注册；
2. **实名认证**（必须做，否则不能租卡；学生证可以用于学生优惠，有就填）；
3. 充值 ¥30（支付宝/微信）。

> 实名通常几分钟内通过。如果卡住，先做第 2~4 步的"看"，等通过再下单。

---

## 2. 选卡：在"算力市场"里怎么点

进入 **算力市场**（<https://www.autodl.com/market/list>），按这个顺序筛：

1. **GPU 型号**：勾 `RTX 4090D`（没货再勾 `RTX 4090`，再没有就 `RTX 3090`）；
2. **区域**：优先 `重庆A区`、`西北B区`、`北京B区`（4090D 库存多）；
3. **看"可租卡数"**：只选 **≥2 卡可租** 的机器——只剩 1 张的机器很容易在你点下去之前被抢走；
4. **镜像**：见下一节，这一步选错要重装，最费时间。

> 抢不到卡是常态。**别在一棵树上等**：换区域、换 4090D、或者降到 3090，都能训。

---

## 3. 镜像怎么选（这一步最容易选错）

创建实例时要选一个"基础镜像"。你要的是：

```
PyTorch  2.5.x   +   Python 3.12   +   CUDA 12.4
```

具体标签类似 `PyTorch / 2.5.1 / 3.12(ubuntu22.04) / cuda12.4`。**只要满足这三条就行**，
`2.4` 或 `2.6` 也可以，**CUDA 必须 12.x**。

| 别选 | 为什么 |
|---|---|
| 社区镜像（别人配好的） | 里面往往有一堆过期包和奇怪的启动脚本，出问题你查不动 |
| 纯 Miniconda 镜像 | 也能用，但要自己装 torch（多 5~10 分钟 + 可能踩 CUDA 版本坑） |
| TensorFlow / 其他框架镜像 | 不是你需要的 |

**为什么强调"不要自己重装 torch"**：镜像里的 torch 已经和 CUDA 对齐了，
你 `pip install torch` 很可能装成 CPU 版或版本不匹配，然后 `torch.cuda.is_available()`
变 False——这是新手最常见的"卡装好了但用不了"。

---

## 4. 磁盘：系统盘 vs 数据盘（搞混会爆盘）

| 路径 | 容量 | 说明 |
|---|---|---|
| `/root/`（系统盘） | 30G 免费 | 放代码可以，**放模型必爆** |
| `/root/autodl-tmp/`（数据盘） | 50G+ 免费 | **模型、数据、HF 缓存都放这里** |
| `/root/autodl-fs/` | 网络存储 | 跨实例共享，慢，P4 用不上 |

创建实例时把**数据盘拉到 50G**（一般免费额度够）。所有大文件都进 `autodl-tmp`。

---

## 5. 计费方式 + 第一个关键操作：**无卡模式**

- 计费选 **按量计费**；
- **不要直接"有卡开机"**。先选 **无卡模式开机**（约 ¥0.1/h，没有 GPU 但 CPU/磁盘/网络都在）。

为什么：传文件、装依赖、下模型这三件事**不需要 GPU**。
用 GPU 模式做这些，等于花 ¥1.9/h 等 `pip` 下载。无卡模式做完，再关机切成 GPU 模式。

> 界面位置：创建实例后，实例卡片上 **"开机"旁边的下拉/更多菜单 → 无卡模式开机**。
> 如果没有这个选项，说明该区域不支持，直接有卡开机也行，只是多花几块钱。

---

## 6. 创建实例：每个字段的意思

点"创建"时会看到这些字段，逐个解释：

| 字段 | 你填什么 | 说明 |
|---|---|---|
| 主机 | 选定的那台 | 看"可租卡数 ≥2" |
| GPU 数量 | **1** | P4 单卡足够，多卡不加分只加钱 |
| 镜像 | PyTorch 2.5 + cu124 | 见第 3 节 |
| 数据盘 | 50G | 见第 4 节 |
| 计费方式 | 按量 | |
| 实例名称 | `p4-sqlrl` | 方便你在一堆实例里认出来 |

创建后约 30 秒启动。**从这一刻开始计费**（无卡模式按无卡价）。

---

## 7. 连上机器：两种方式，选一种

### 方式 A：JupyterLab（最省事，适合传文件）

实例卡片 → **JupyterLab** → 浏览器打开一个网页版 IDE。
左侧文件树可以**直接拖拽上传** `p4_bundle.zip`，还能开 Terminal 敲命令。

### 方式 B：SSH（推荐，稳定，适合长任务）

在实例详情页找到连接信息，形如：

```
ssh -p 12345 root@connect.cqa1.seetacloud.com
```

在 **Windows PowerShell** 里直接粘贴这行（Windows 10/11 自带 ssh）：

```powershell
ssh -p 12345 root@connect.cqa1.seetacloud.com
```

第一次连接会问 `yes/no`，输入 `yes`；然后输入密码（**粘贴密码时屏幕上不会显示任何字符**，
这是正常的，粘完直接回车）。

> 密码在实例详情页。复制时注意别多复制空格；含特殊字符时用单引号包起来。

---

## 8. 连上之后，先做这 4 个检查

```bash
pwd                 # 应该是 /root
df -h /root /root/autodl-tmp   # 看两个盘各剩多少
nvidia-smi          # 无卡模式下会报错，这是正常的；GPU 模式下要能看到 4090D
cat /etc/os-release | head -2
```

**无卡模式下 `nvidia-smi` 报 `NVIDIA-SMI has failed` 是预期行为**，不要以为机器坏了。

---

## 9. 传文件 + 装环境（还在无卡模式）

### 9.1 传 zip

JupyterLab 直接拖到 `/root/autodl-tmp/`；或者本地 PowerShell：

```powershell
scp -P 12345 "E:\deepseek工作区\sql-agent-rl\dist\p4_bundle.zip" root@connect.cqa1.seetacloud.com:/root/autodl-tmp/
```

### 9.2 解压 + 装依赖 + 下模型

```bash
cd /root/autodl-tmp
unzip -q p4_bundle.zip -d sql-agent-rl
cd sql-agent-rl
ls            # 应看到 training/ scripts/ envs/ configs/ guides/ data/

# 网络加速（AutoDL 内置，加速 GitHub 和 HuggingFace）
source /etc/network_turbo

export HF_HOME=/root/autodl-tmp/hf
echo 'export HF_HOME=/root/autodl-tmp/hf' >> ~/.bashrc

pip install -r requirements-train.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下模型（约 8G，5~10 分钟）
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B-Instruct-2507")
print("模型已就绪")
PY
du -sh /root/autodl-tmp/hf
```

> `source /etc/network_turbo` 只对**当前这个终端**有效；重连后要再执行一次。
> 如果 HF 还是慢，再加一句 `export HF_ENDPOINT=https://hf-mirror.com`。
> **装完检查**：`python -c "import torch;print(torch.__version__, torch.cuda.is_available())"` ——
> 无卡模式下 `False` 正常，GPU 模式下必须是 `True`。

---

## 10. 切 GPU 模式，开始 P4

1. 实例卡片 → **关机**；
2. 关机后 → **开机（GPU 模式）**；
3. 重新 SSH 进去，`nvidia-smi` 应看到 4090D/24G；
4. 回到 `guides/P4.md` 的 **Step 5**，从 `python scripts/build_p4_data.py` 开始。

```bash
cd /root/autodl-tmp/sql-agent-rl
export HF_HOME=/root/autodl-tmp/hf
python scripts/build_p4_data.py
python scripts/check_p4_env.py        # ← 到这里停下来，把输出贴给我
```

---

## 11. 长任务必须用 tmux（否则 SSH 一断，30 分钟白跑）

SFT 要跑 30~50 分钟。**SSH 连接断了，前台进程会被杀掉**。所以：

```bash
# 装一次（一般已自带）
apt-get install -y tmux

# 开一个会话
tmux new -s sft

# 在里面跑训练
python -m training.train_sft --config configs/train/sft_qwen3_4b.json

# 中途想离开：按 Ctrl+B，松手，再按 D（detach）
# 回来继续看：
tmux attach -t sft

# 看会话列表
tmux ls
```

或者用 `nohup`（不能交互，但够用）：

```bash
nohup python -m training.train_sft --config configs/train/sft_qwen3_4b.json > sft.log 2>&1 &
tail -f sft.log
```

---

## 12. 关机 vs 释放（这条最贵）

| 操作 | 计费 | 数据 | 什么时候用 |
|---|---|---|---|
| **关机** | 停 | **保留 15 天** | 每次跑完都做这个 |
| 释放 | 停 | **立刻全删** | 项目彻底结束、产物已拉回本地时 |

- 关机后**数据盘保留 15 天不收费**，所以不用急着拉走，但别拖过半个月；
- **释放是不可逆的**。点之前先确认 `runs/` 和 `results/` 已经拉到本地。

---

## 13. 新手最常踩的 7 个坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 没抢到卡 | 创建按钮灰的 / 提示库存不足 | 换区域、换 4090D、换 3090；或等 10 分钟 |
| 模型下到系统盘 | 提示 `No space left on device` | 先 `export HF_HOME=/root/autodl-tmp/hf` 再下；`df -h` 自查 |
| 自己重装了 torch | `torch.cuda.is_available()` 是 False | 别重装；`pip install ... --no-deps` 补其它包 |
| SSH 断开任务死 | 训练日志停在半路 | 用 tmux / nohup（第 11 节） |
| 密码粘贴失败 | 反复 `Permission denied` | 密码框不显示字符是正常的；确认没多复制空格 |
| 代理开太久 | 访问某些站点 503 | `unset http_proxy https_proxy` 后再试 |
| 忘记关机 | 余额半夜烧光 | 跑完立刻关机；手机端也能关机 |

---

## 14. 这一步的完成标准（发我这三行就够）

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__,torch.cuda.is_available())"
ls /root/autodl-tmp/sql-agent-rl
```

三行都对得上（4090D/24G、`True`、仓库目录列出来），就算连通了。
**不要发我密码**——所有命令你自己敲，把输出贴回来就行。
