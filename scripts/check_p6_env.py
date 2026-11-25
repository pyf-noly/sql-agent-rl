#!/usr/bin/env python3
"""P6 上卡后的第一条命令：环境 / BIRD 数据 / 数据库 / 模型 / 模板 / 一次 rollout 全验。

租来的机器按小时计费，"训练跑到第 200 步才报错"是最贵的失败方式。
这个脚本 2~3 分钟把**会在中途才暴露**的问题提前打出来：

1. 显卡、bf16、显存、磁盘（4B + LoRA + 3072 序列需要 ~20G 显存与 ~20G 磁盘）；
2. BIRD 数据在不在、条数对不对、`meta.json` 的自检是否通过；
3. 三套 sqlite：dev（评测）/ train（SFT 不直接用，但 GRPO 的奖励要）/ RL 留出库是否齐；
4. chat 模板 mask 边界（transformers 4.x / 5.x 都要对）；
5. 一次生成 + 一次完整 rollout + 判分（用留出库的真实样本）。

用法::

    python scripts/check_p6_env.py                 # 全量（含加载模型，2~3 分钟）
    python scripts/check_p6_env.py --skip-model    # 只看环境与数据（5 秒）
    python scripts/check_p6_env.py --model /root/autodl-tmp/qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "✓", "✗", "!"


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "", *, hard: bool = True) -> None:
        self.items.append({"name": name, "ok": ok, "detail": detail, "hard": hard})
        mark = OK if ok else (WARN if not hard else FAIL)
        print(f"  [{mark}] {name:<30} {detail}")

    @property
    def failed_hard(self) -> list[dict]:
        return [i for i in self.items if not i["ok"] and i["hard"]]


def _count_lines(p: Path) -> int:
    if not p.is_file():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/qwen3-4b")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--data-dir", default="data/bird")
    ap.add_argument("--bird-root", default="data/raw/bird")
    args = ap.parse_args()

    rep = Report()
    t0 = time.time()
    data_dir = ROOT / args.data_dir
    bird_root = ROOT / args.bird_root
    dev_db_root = bird_root / "dev_20240627/dev_databases"
    train_db_root = bird_root / "train/train_databases"

    # ---------------- 1. 环境 ----------------
    print("\n== 1. 环境 ==")
    try:
        import torch
    except ImportError:
        rep.add("import torch", False, "没装 torch：先按 guides/P6.md 装依赖")
        return _finish(rep, t0, ROOT)
    import platform

    rep.add("python", True, platform.python_version())
    rep.add("torch", True, torch.__version__)
    rep.add("cuda 可用", torch.cuda.is_available(),
            "" if torch.cuda.is_available() else "镜像选错或驱动没起来")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 2**30
        rep.add("gpu", True, f"{props.name} {vram:.1f}G")
        rep.add("显存 >= 22G", vram >= 22, f"{vram:.1f}G（3072 序列 + LoRA 需要）",
                hard=False)
        rep.add("bf16 支持", torch.cuda.is_bf16_supported(), "", hard=False)
        used = torch.cuda.memory_allocated() / 2**30
        rep.add("卡上没有残留进程", used < 0.5, f"已分配 {used:.2f}G（开训前必须接近 0）")

    for mod in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            m = __import__(mod)
            rep.add(mod, True, getattr(m, "__version__", "?"))
        except ImportError:
            rep.add(mod, False, "未安装（bitsandbytes 只有本地 4-bit 评测才需要）",
                    hard=(mod != "bitsandbytes"))

    free = shutil.disk_usage(ROOT).free / 2**30
    rep.add("磁盘剩余 >= 15G", free > 15, f"{free:.0f}G", hard=False)

    # ---------------- 2. 数据 ----------------
    print("\n== 2. BIRD 数据 ==")
    for name, expect in (("sft_train.jsonl", 1000), ("sft_val.jsonl", 20),
                         ("rl_train.jsonl", 500), ("dev.json", 1000)):
        n = _count_lines(data_dir / name)
        rep.add(f"{args.data_dir}/{name}", n >= expect, f"{n} 条")

    meta_p = data_dir / "meta.json"
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        checks = meta.get("checks", {})
        rep.add("数据自检（prepare_bird）", bool(checks.get("ok")),
                "通过" if checks.get("ok") else str(checks.get("failures"))[:140])
        tr = meta.get("train", {})
        rep.add("库级切分无重叠", tr.get("sft_rl_prompt_overlap") == 0,
                f"SFT/RL 重叠 {tr.get('sft_rl_prompt_overlap')}，"
                f"train/dev 库重叠 {tr.get('train_dev_db_overlap')}")
    else:
        rep.add("数据自检（prepare_bird）", False, "缺 meta.json：先跑 prepare_bird.py")

    # ---------------- 3. sqlite ----------------
    print("\n== 3. 数据库 ==")
    n_dev = len(list(dev_db_root.glob("*/*.sqlite")))
    rep.add("dev 数据库（评测必须）", n_dev >= 11, f"{n_dev} 个（期望 11）")
    n_train = len(list(train_db_root.glob("*/*.sqlite")))
    rep.add("train 数据库（SFT 不需要；GRPO 奖励要）", n_train >= 1,
            f"{n_train} 个（完整 69；只下 RL 留出库也可以）", hard=False)
    split_p = data_dir / "db_split.json"
    if split_p.is_file():
        split = json.loads(split_p.read_text(encoding="utf-8"))
        rl_dbs = split.get("rl_dbs", [])
        missing = [d for d in rl_dbs
                   if not (train_db_root / d / f"{d}.sqlite").is_file()]
        rep.add("RL 留出库齐全", not missing,
                f"{len(rl_dbs) - len(missing)}/{len(rl_dbs)}"
                + (f"，缺 {missing[:3]}" if missing else ""))

    # ---------------- 4. 模型与模板 ----------------
    if args.skip_model:
        print("\n== 4. 模型与模板：跳过（--skip-model）==")
        return _finish(rep, t0, ROOT)

    print("\n== 4. 模型与模板 ==")
    mp = Path(args.model)
    if not mp.exists():
        rep.add("模型目录", False, f"{mp} 不存在：先跑 scripts/download_model_ms.py")
        return _finish(rep, t0, ROOT)
    weights = list(mp.glob("*.safetensors"))
    size_gb = sum(p.stat().st_size for p in weights) / 1e9 if weights else 0
    rep.add("模型权重", size_gb > 6, f"{len(weights)} 个文件 {size_gb:.1f}GB")

    from training.chatfmt import encode_with_spans
    from training.config import ModelConfig
    from training.model import count_params, load_base

    try:
        model, tok = load_base(str(mp), ModelConfig(name=str(mp)))
    except Exception as e:  # noqa: BLE001
        rep.add("加载模型", False, f"{type(e).__name__}: {e}")
        return _finish(rep, t0, ROOT)
    rep.add("加载模型", True, str(mp))
    info = count_params(model)
    rep.add("参数量", True, f"{info['params_total_b']}B")

    rl_path = data_dir / "rl_train.jsonl"
    if not rl_path.is_file():
        print("  [跳过] 没有 rl_train.jsonl，无法做模板检查")
        return _finish(rep, t0, ROOT)
    rows = [json.loads(line) for line in rl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    sample = rows[0]
    conversation = list(sample["prompt_messages"]) + [
        {"role": "assistant", "content": "```sql\nSELECT COUNT(*) FROM t;\n```"}]
    enc = encode_with_spans(tok, conversation, max_len=3072)
    rep.add("chat 模板 mask（3072）", enc.n_mismatch == 0 and enc.n_loss_tokens > 5,
            f"长度 {enc.length}，loss {enc.n_loss_tokens}，"
            f"上下文 {enc.context_ratio:.0%}，mismatch {enc.n_mismatch}")
    rep.add("prompt 里带 evidence/类型/引号",
            "### External knowledge" in sample["prompt_messages"][1]["content"]
            or "CREATE TABLE" in sample["prompt_messages"][1]["content"],
            "BIRD prompt 形态", hard=False)

    # ---------------- 5. 生成 + rollout ----------------
    print("\n== 5. 生成 + rollout ==")
    from rollout.parse_sql import parse_sql
    from training.model import generate_text

    text = generate_text(model, tok, sample["prompt_messages"],
                         max_new_tokens=64, temperature=0.0)
    parsed = parse_sql(text)
    rep.add("生成 + 解析", parsed.ok,
            f"抽到 SQL {len(parsed.sql)} 字符" if parsed.ok else f"原文 {text[:80]!r}")

    if not args.skip_rollout:
        from envs.db_registry import DatabaseRegistry
        from training.rollout import HFPolicyClient, episode_reward, make_agent

        dev_rows = json.loads((data_dir / "dev.json").read_text(encoding="utf-8"))
        registry = DatabaseRegistry(dev_db_root)
        client = HFPolicyClient(model, tok, max_new_tokens=192, temperature=0.0)
        agent, ev = make_agent(client, registry, max_turns=1, timeout=10.0,
                               temperature=0.0, max_rows=20000, compare="bird")
        ep = agent.run(dev_rows[0])
        r, info = episode_reward(ep, dev_rows[0], ev)
        rep.add("rollout", ep.n_turns >= 1, f"轮数 {ep.n_turns}，停止原因 {ep.stopped_reason}")
        rep.add("判分", True, f"reward={r} correct={info['correct']} reason={info['reason']}")

    return _finish(rep, t0, ROOT)


def _finish(rep: Report, t0: float, root: Path) -> int:
    out = root / "runs/p6_check"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_s": round(time.time() - t0, 1),
        "n_checks": len(rep.items),
        "failed": [i for i in rep.items if not i["ok"]],
        "items": rep.items,
    }
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    hard = rep.failed_hard
    print("\n" + "=" * 60)
    if hard:
        print(f"结论：{len(hard)} 项硬性检查未通过，先修再训练：")
        for i in hard:
            print(f"  - {i['name']}: {i['detail']}")
        print("=" * 60)
        return 1
    print(f"结论：全部硬性检查通过，耗时 {payload['elapsed_s']}s。可以开训。")
    print("报告：runs/p6_check/report.json")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
