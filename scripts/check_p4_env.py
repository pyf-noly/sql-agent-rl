#!/usr/bin/env python3
"""P4 上卡后的第一条命令：把"环境/数据/模型/模板"四件事一次性验完。

为什么要单独一个脚本
-------------------
租来的机器按小时计费，"训练跑到第 200 步才报错"是最贵的失败方式。
这个脚本在 2~3 分钟内把下面这些**会在训练中途才暴露**的问题提前打出来：

1. 显卡是不是真的能用（bf16、显存、驱动）；
2. 数据在不在、格式对不对（P3 产物有没有同步上来）；
3. chat 模板在这台机器的 transformers 版本上，mask 边界对不对；
4. 模型能不能加载、生不生成得出 ```sql 块；
5. 一条完整 rollout + 判分能不能跑通。

用法::

    python scripts/check_p4_env.py                  # 全量（含加载模型，约 2~3 分钟）
    python scripts/check_p4_env.py --skip-model     # 只看环境与数据（10 秒）
    python scripts/check_p4_env.py --model Qwen/Qwen3-1.7B
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

    def add(self, name: str, ok: bool, detail: str = "", *, hard: bool = True,
            warn: bool = False) -> None:
        self.items.append({"name": name, "ok": ok, "detail": detail,
                           "hard": hard, "warn": warn})
        mark = OK if ok else (WARN if warn else FAIL)
        print(f"  [{mark}] {name:<28} {detail}")

    @property
    def failed_hard(self) -> list[dict]:
        return [i for i in self.items if not i["ok"] and i["hard"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-rollout", action="store_true")
    args = ap.parse_args()

    rep = Report()
    t0 = time.time()

    # ---------------- 1. 环境 ----------------
    print("\n== 1. 环境 ==")
    try:
        import torch
    except ImportError:
        rep.add("import torch", False, "没装 torch：先按 guides/P4.md 装依赖")
        return _finish(rep, t0, ROOT)
    import platform

    py = platform.python_version()
    rep.add("python", True, py)
    rep.add("torch", True, torch.__version__)
    rep.add("cuda 可用", torch.cuda.is_available(),
            "" if torch.cuda.is_available() else "没检测到 CUDA：镜像选错了，或驱动没起来")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 2**30
        rep.add("gpu", True, f"{props.name} {vram:.1f}G")
        rep.add("显存 >= 20G", vram >= 20, f"{vram:.1f}G（4B+LoRA 建议 24G）", hard=False, warn=True)
        rep.add("bf16 支持", torch.cuda.is_bf16_supported(),
                "" if torch.cuda.is_bf16_supported() else "会降级 fp16，慢一点但不致命",
                hard=False, warn=True)

    for mod in ("transformers", "peft", "accelerate"):
        try:
            m = __import__(mod)
            rep.add(mod, True, getattr(m, "__version__", "?"))
        except ImportError:
            rep.add(mod, False, "未安装")

    free = shutil.disk_usage(ROOT).free / 2**30
    rep.add("磁盘剩余", free > 20, f"{free:.0f}G（模型权重 + adapter + 日志）", hard=False, warn=True)

    # ---------------- 2. 数据 ----------------
    print("\n== 2. 数据 ==")
    d = ROOT / "data/train"
    for name, expect in (("rl_train.jsonl", 100), ("sft_train.jsonl", 100), ("sft_val.jsonl", 20)):
        p = d / name
        n = 0
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                n = sum(1 for _ in f)
        rep.add(f"data/train/{name}", p.is_file() and n >= expect, f"{n} 条")

    meta_p = d / "train_data.meta.json"
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        ok = bool(meta.get("checks", {}).get("ok"))
        rep.add("数据自检（build_p4_data）", ok,
                "通过" if ok else str(meta.get("checks", {}).get("failures"))[:120])
    else:
        rep.add("数据自检（build_p4_data）", False, "缺 meta：先跑 build_p4_data.py")

    db_root = ROOT / "data/raw/spider/database"
    n_db = len(list(db_root.glob("*/*.sqlite"))) if db_root.is_dir() else 0
    rep.add("spider 数据库", n_db >= 150, f"{n_db} 个")

    # ---------------- 3. 模型与模板 ----------------
    if args.skip_model:
        print("\n== 3. 模型与模板：跳过（--skip-model）==")
        return _finish(rep, t0, ROOT)

    print("\n== 3. 模型与模板 ==")
    from training.chatfmt import encode_with_spans
    from training.model import count_params, load_base

    try:
        model, tok = load_base(args.model, _model_cfg(args.model))
    except Exception as e:  # noqa: BLE001
        rep.add("加载模型", False, f"{type(e).__name__}: {e}（检查 HF 镜像/磁盘）")
        return _finish(rep, t0, ROOT)
    rep.add("加载模型", True, args.model)
    info = count_params(model)
    rep.add("参数量", True, f"{info['params_total_b']}B")

    rows = [json.loads(line) for line in
            (d / "rl_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = rows[0]
    conversation = list(sample["prompt_messages"]) + [
        {"role": "assistant", "content": "```sql\nSELECT COUNT(*) FROM student;\n```"}]
    enc = encode_with_spans(tok, conversation, max_len=2048)
    rep.add("chat 模板 mask", enc.n_mismatch == 0 and enc.n_loss_tokens > 5,
            f"长度 {enc.length}，loss token {enc.n_loss_tokens}，"
            f"上下文占比 {enc.context_ratio:.0%}，mismatch {enc.n_mismatch}")

    # ---------------- 4. 一次生成 ----------------
    print("\n== 4. 生成 ==")
    from rollout.parse_sql import parse_sql
    from training.model import generate_text

    text = generate_text(model, tok, sample["prompt_messages"],
                         max_new_tokens=64, temperature=0.0)
    parsed = parse_sql(text)
    rep.add("生成 + 解析", parsed.ok, f"抽到 SQL {len(parsed.sql)} 字符" if parsed.ok
            else f"没抽出 SQL，原文：{text[:80]!r}")
    print(f"    输出预览：{text[:160].replace(chr(10), ' ')}")

    # ---------------- 5. 一条 rollout ----------------
    if not args.skip_rollout:
        print("\n== 5. 单条 rollout + 判分 ==")
        from envs.db_registry import DatabaseRegistry
        from training.rollout import HFPolicyClient, episode_reward, make_agent

        dev = json.loads((ROOT / "data/processed/dev_sample.json").read_text(encoding="utf-8"))
        registry = DatabaseRegistry(db_root)
        client = HFPolicyClient(model, tok, max_new_tokens=128, temperature=0.0)
        agent, ev = make_agent(client, registry, max_turns=1, timeout=2.0, temperature=0.0)
        ep = agent.run(dev[0])
        r, info = episode_reward(ep, dev[0], ev)
        rep.add("rollout", ep.n_turns >= 1, f"轮数 {ep.n_turns}，停止原因 {ep.stopped_reason}")
        rep.add("判分", True, f"reward={r} correct={info['correct']} reason={info['reason']}")

    return _finish(rep, t0, ROOT)


def _model_cfg(name: str):
    from training.config import ModelConfig

    return ModelConfig(name=name)


def _finish(rep: Report, t0: float, root: Path) -> int:
    out = root / "runs/p4_check"
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
    print("\n" + "=" * 56)
    if hard:
        print(f"结论：{len(hard)} 项硬性检查未通过，先修再训练：")
        for i in hard:
            print(f"  - {i['name']}: {i['detail']}")
        print("=" * 56)
        return 1
    print(f"结论：全部硬性检查通过，耗时 {payload['elapsed_s']}s。可以开训。")
    print(f"报告：runs/p4_check/report.json")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
