# B-lite 执行手册：vLLM 服务化评测的吞吐实验

> 目标：回答「换成 vLLM 服务后，评测吞吐能到多少、延迟 p95 怎么变、每千题成本多少」。
> **不改训练代码**：只把 `run_baseline.py --backend api` 指到 vLLM，量并发吞吐。
> 对照基线（HF 单卡串行，1534 条）：**23.4 题/分，p50 2.44s，p95 4.09s，¥1.78/千题**。

---

## 0. 前置检查（开机后第一条命令）

```bash
nvidia-smi --query-gpu=name,memory.used --format=csv     # 必须 0 MiB
cd /root/autodl-tmp/p6/sql-agent-rl && ls runs/sft_bird_p6v3/adapter/adapter_model.safetensors
ls data/raw/bird/dev_20240627/dev_databases | wc -l      # 应该 11
df -h /root/autodl-tmp | tail -1                          # 至少留 10G
```

数据盘保留 15 天（到 **11-21 左右**）；过期就要重传 bundle + 重下模型。

---

## 1. 装 vLLM（**单独 venv，别动训练环境**）

```bash
python -m venv /root/vllm-env
/root/vllm-env/bin/pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# Qwen3-4B-Instruct-2507 需要较新的 vLLM；先装最新版
/root/vllm-env/bin/pip install -U vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
/root/vllm-env/bin/python -c "import vllm; print(vllm.__version__)"
```

**风险点**：vLLM 会拉自己的 torch/CUDA 依赖。装到独立 venv 里就不会破坏 `torch 2.5.1 + transformers 4.57` 的训练环境。

如果 vLLM 装不上（CUDA 不匹配、wheel 缺失），备选：
1. `sglang[all]`（同样 OpenAI 兼容，`python -m sglang.launch_server`）；
2. 实在不行，用 `lmdeploy`；
3. 最后手段：只测 HF + 多进程服务（`--backend api` + 本地开多个 `run_baseline`），但这不是 B-lite 想要的结论。

---

## 2. 起服务（nohup，带 LoRA）

```bash
cd /root/autodl-tmp/p6/sql-agent-rl
nohup /root/vllm-env/bin/vllm serve /root/autodl-tmp/qwen3-4b \
  --served-model-name qwen3-4b \
  --enable-lora \
  --lora-modules sft=/root/autodl-tmp/p6/sql-agent-rl/runs/sft_bird_p6v3/adapter \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  > /root/vllm.log 2>&1 &
```

等它就绪（约 1~3 分钟）：

```bash
until curl -s http://127.0.0.1:8000/v1/models >/dev/null; do sleep 3; done
curl -s http://127.0.0.1:8000/v1/models | head -c 400
```

**如果 LoRA 加载失败**（`--lora-modules` 报错）：先去掉这两行、只服务基座模型，B-lite 只比吞吐，结论仍然成立（EA 会低，注明即可）。

---

## 3. 冒烟：先跑 10 条

```bash
cd /root/autodl-tmp/p6/sql-agent-rl
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY OPENAI_MODEL=sft \
python scripts/run_baseline.py --dataset bird --backend api \
  --turns 1 --limit 10 --workers 1 --timeout 10 --max-rows 20000 --value-check \
  --tag blite_smoke
```

看两件事：**能出结果**（不是全 0）、**EA 和 307 条那次同量级**（约 40%~60% 的题上）。

---

## 4. 正式实验：220 条 × 4 档并发

```bash
cd /root/autodl-tmp/p6/sql-agent-rl
python scripts/b_lite_throughput.py \
  --base-url http://127.0.0.1:8000/v1 --model sft \
  --workers 1,4,8,16 --stride 7 --turns 1 --tag-prefix blite
```

- `--stride 7`：1534 → **220 条**，所有并发档跑**同一批题**（可比）；
- 每档预计 1~10 分钟，总共 20~30 分钟；
- 产出 `results/blite_vllm_w{1,4,8,16}.json` + `results/blite_summary.json` + 一张 markdown 表。

如果 16 并发把显存打爆（日志里 `CUDA out of memory`），把 `--workers` 改成 `1,4,8`，并降低 `--gpu-memory-utilization` 重试。

---

## 5. 判读标准

| 看什么 | 合格线 | 说明 |
|---|---|---|
| 吞吐 | 明显高于 **23.4 题/分** | 4~8 并发下到 60~100 题/分算正常 |
| p95 | 不应大幅恶化 | 并发越高 p95 越长是正常的，但要 record |
| EA | 与 307 条那次同口径**一致**（±2pp 内） | 差太多说明 prompt/模板/模型名配错了 |
| 成本 | 用 GPU 墙钟算，别用 `usage.cost_cny` | vLLM 的 token 计数会被按 DeepSeek 价目误算 |

**注意**：`results/*.json` 里的 `cost_cny` 是 `llm_client` 按 DeepSeek 单价算的，
对 vLLM 没有意义；成本一律用 `scripts/latency_report.py --gpu-price 2.5` 重算。

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `vllm: command not found` | venv 没激活 | 用 `/root/vllm-env/bin/vllm` 全路径 |
| 启动报 `does not support architecture Qwen3ForCausalLM` | vLLM 太老 | 升级 vLLM；或换 SGLang |
| `--lora-modules` 报 adapter 不兼容 | peft 版本/rank 配置 | 去掉 LoRA 只服务基座，注明 |
| 请求报 `model not found` | `OPENAI_MODEL` 与 `--served-model-name`/LoRA 名不一致 | 改成 `sft` 或 `qwen3-4b` |
| 全部超时 | 并发太高、KV cache 不够 | 降 workers / 降 `--gpu-memory-utilization` / 降 `max-model-len` |
| 结果全错 | chat template 没对齐 | 先用基座模型 + 单条 curl 看输出格式 |

---

## 7. 收尾

```bash
# 拉回本地（在本地跑）
scp -P <端口> root@<主机>:/root/autodl-tmp/p6/sql-agent-rl/results/blite_vllm_w*.json  E:\deepseek工作区\sql-agent-rl\results\
scp -P <端口> root@<主机>:/root/autodl-tmp/p6/sql-agent-rl/results/blite_summary.json   E:\deepseek工作区\sql-agent-rl\results\

# 关机（控制台）
```

跑完把 `blite_summary.json` 给我，我把它并进 `docs/serving_metrics.md`，并把「同步 vs 服务化」写成一段可讲的结论。

---

## 8. 这个实验能讲什么、不能讲什么

**能讲**：
- 服务化后吞吐从 23.4 题/分提到 X，p95 从 4.09s 变 Y；
- 并发到 8 之后开始饱和（如果观察到），说明瓶颈在 GPU 算力而不是调度；
- 每千题成本（GPU 口径）随并发下降。

**不能讲**：
- 这不是 RL 训练时的 rollout 吞吐（那是 B-full，要改训练循环 + 权重同步）；
- vLLM 的 continuous batching 和训练时的异步 rollout 是两件事，面试别混。
