#!/usr/bin/env python3
"""单题多轮演示：把 Agent 每一轮「生成 → 执行 → 反馈」和耗时直接打出来。

用途
----
1. 面试/演示：一条命令看到「模型写错 → 环境反馈 → 模型改对」的完整过程；
2. 调试：不用跑整批评测，只看一道题为什么错；
3. 服务化：顺带给出单题延迟与 token 量（``--json`` 可落盘）。

用法::

    # 本地 8G 显卡（4-bit）
    python scripts/demo_agent.py --index 5 --turns 3 --value-check \
        --backend hf --model-path ../.models/Qwen3-4B-Instruct-2507 --load-in-4bit

    # 用 OpenAI 兼容接口（DeepSeek / vLLM / SGLang）
    python scripts/demo_agent.py --backend api --index 5 --turns 3

    # 自己给一道题（没有 gold，跳过判分）
    python scripts/demo_agent.py --question "有多少学生？" --db california_schools

    # 列前 10 条样本
    python scripts/demo_agent.py --list 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.db_registry import DatabaseRegistry  # noqa: E402
from rollout.agent import SqlAgent  # noqa: E402
from rollout.evaluator import Evaluator  # noqa: E402
from scripts.run_baseline import DATASETS  # noqa: E402  （复用数据集路径，避免两套配置漂移）


class TimedClient:
    """包一层计时：记录每次 ``chat`` 的耗时，同时透明转发其他属性。"""

    def __init__(self, inner):
        self.inner = inner
        self.times: list[float] = []
        self.usage = getattr(inner, "usage", None)

    @property
    def model_name(self):
        return getattr(self.inner, "model_name", "local")

    def chat(self, messages, **kw):
        t0 = time.perf_counter()
        try:
            return self.inner.chat(messages, **kw)
        finally:
            self.times.append(time.perf_counter() - t0)


def build_client(args):
    if args.backend == "hf":
        from training.config import ModelConfig
        from training.model import load_for_eval
        from training.rollout import HFPolicyClient

        print(f"[模型] 本地 {args.model_path}" + ("（4-bit）" if args.load_in_4bit else "")
              + (f" + adapter {args.adapter}" if args.adapter else ""))
        t0 = time.perf_counter()
        mc = ModelConfig(name=args.model_path, load_in_4bit=args.load_in_4bit)
        model, tok = load_for_eval(mc.name, mc, adapter=args.adapter)
        print(f"[模型] 加载完成 {time.perf_counter() - t0:.1f}s")
        return HFPolicyClient(model, tok, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, name=mc.name)

    from rollout.llm_client import make_client_from_env
    return make_client_from_env(temperature=args.temperature, max_tokens=args.max_new_tokens)


def pick_sample(args, rows: list[dict]) -> dict:
    if args.question:
        return {"db_id": args.db, "question": args.question,
                "query": "", "evidence": args.evidence or ""}
    if args.index is None:
        args.index = 0
    if args.index >= len(rows):
        raise SystemExit(f"--index {args.index} 超出范围（共 {len(rows)} 条）")
    return rows[args.index]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="bird")
    ap.add_argument("--db-root", default="", help="覆盖数据集默认的 sqlite 根目录")
    ap.add_argument("--index", type=int, default=None, help="取第几条样本（0-based）")
    ap.add_argument("--list", type=int, default=0, help="只列前 N 条样本，不跑模型")
    ap.add_argument("--question", default="", help="自定义问题（需要同时给 --db）")
    ap.add_argument("--db", default="", help="配合 --question 的 db_id")
    ap.add_argument("--evidence", default="", help="BIRD 的外部知识提示（可选）")
    ap.add_argument("--turns", type=int, default=3, help="最多几轮，默认 3（判对才停）")
    ap.add_argument("--backend", choices=["hf", "api"], default="hf")
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = 用数据集默认")
    ap.add_argument("--value-check", action="store_true", help="执行成功时检查取值是否存在")
    ap.add_argument("--full-feedback", action="store_true", help="完整打印反馈（默认截断 400 字）")
    ap.add_argument("--json", default="", help="把这一条 Episode 写成 JSON")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    data_path = ROOT / ds["data"]
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    print(f"[数据] {data_path.relative_to(ROOT)}：{len(rows)} 条")

    if args.list:
        for i, r in enumerate(rows[: args.list]):
            print(f"  [{i}] {r['db_id']:24s} {r['question'][:70]}")
        return 0

    sample = pick_sample(args, rows)
    db_id = sample["db_id"]
    gold = sample.get("query", "")
    print(f"\n[问题] {sample['question']}")
    print(f"[数据库] {db_id}")
    if sample.get("evidence"):
        print(f"[外部知识] {sample['evidence'][:200]}")

    registry = DatabaseRegistry(ROOT / (args.db_root or ds["db_root"]))
    max_rows = args.max_rows or ds["max_rows"]
    compare = ds["compare"]
    timeout = args.timeout or ds.get("timeout", 10.0)

    ev = Evaluator(registry, timeout=timeout, max_rows=max_rows, compare=compare)
    inner = build_client(args)
    client = TimedClient(inner)

    agent = SqlAgent(
        client, registry,
        max_turns=args.turns,
        timeout=timeout,
        temperature=args.temperature,
        stop_on_success=False,
        stop_on_correct=True,     # 演示「判对才停」
        verifier=ev if gold else None,
        prompt_cfg=ds["prompt"],
        max_rows=max_rows,
        value_check=args.value_check,
    )

    t0 = time.perf_counter()
    ep = agent.run(sample)
    wall = time.perf_counter() - t0

    for i, turn in enumerate(ep.turns):
        dur = client.times[i] if i < len(client.times) else 0.0
        print(f"\n[第 {turn.turn} 轮] 生成 {dur:.2f}s")
        print(f"  SQL: {turn.sql or '(没有解析出 SQL)'}")
        status = "执行成功" if turn.success else f"执行失败（{turn.error_class}）"
        print(f"  结果: {status}" + (f"，{turn.row_count} 行" if turn.success else ""))
        fb = turn.feedback if args.full_feedback else turn.feedback[:400]
        if fb:
            print("  反馈: " + fb.replace("\n", "\n        "))

    print("\n[判定] ", end="")
    if not gold:
        print("（没有 gold，跳过判分）")
    elif not ep.final_sql:
        print(f"✗ 没有可执行的 SQL（{ep.stopped_reason}）")
    else:
        res = ev.evaluate(db_id, gold, ep.final_sql)
        print(("✓ 正确" if res.correct else f"✗ 错误（{res.reason}）"))

    usage = {}
    if getattr(inner, "usage", None) is not None:
        try:
            usage = inner.usage.to_dict()
        except Exception:  # noqa: BLE001
            usage = {}
    print(f"[汇总] {ep.n_turns} 轮 | 墙钟 {wall:.2f}s | "
          f"生成合计 {sum(client.times):.2f}s | "
          f"prompt {usage.get('prompt_tokens', 0)} tok / "
          f"completion {usage.get('completion_tokens', 0)} tok（估算）| "
          f"停止原因 {ep.stopped_reason}")

    if args.json:
        d = ep.to_dict(truncate_feedback=not args.full_feedback)
        d["wall_sec"] = round(wall, 3)
        d["turn_latency_sec"] = [round(x, 3) for x in client.times]
        d["usage"] = usage
        Path(args.json).write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[写出] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
