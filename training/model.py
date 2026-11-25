"""模型装载与生成：HF + peft(LoRA)，单卡，不依赖 vLLM。

为什么不用 verl / vLLM 起手（重要决策，详见 ``docs/p4_decision.md``）
------------------------------------------------------------------
P4 的目标是"跑通并看懂"，不是"跑得快"。verl 要装 Ray + vLLM + flash-attn，
版本矩阵对一台临时租来的 4090 很不友好；而 P6 的轮级优势本来就要改 advantage
计算，框架越重改起来越疼。所以这里用最薄的实现：

    transformers(前向/反向) + peft(LoRA) + 自己的 rollout 循环

代价是生成慢（HF generate 比 vLLM 慢 2~4 倍），收益是**任何一行出错你都能读**。
等 P5 要跑 300 步时再决定要不要换 vLLM 生成（接口已经隔离在 ``generate_text``）。
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from training.chatfmt import apply_template
from training.config import ModelConfig

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def pick_dtype(name: str) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("[警告] 显卡不支持 bf16，自动降级为 float16")
        return torch.float16
    return _DTYPES.get(name, torch.bfloat16)


def load_tokenizer(name: str, *, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(name: str, cfg: ModelConfig):
    """加载基座（不套 LoRA），返回 ``(model, tokenizer)``。

    ``cfg.load_in_4bit=True`` 时用 bitsandbytes 的 nf4 量化装载，并交给
    ``device_map="auto"`` 放卡——**不要**再 ``.to("cuda")``，量化模型整体搬运
    会报错或把显存打爆。8G 的 4060 靠这条路径跑 4B 推理。
    """
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(name, trust_remote_code=cfg.trust_remote_code)
    kwargs: dict[str, Any] = {
        "attn_implementation": cfg.attn,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pick_dtype(cfg.dtype),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=pick_dtype(cfg.dtype), **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=pick_dtype(cfg.dtype), **kwargs)
        model.config.pad_token_id = tok.pad_token_id
        model.eval()
        return model, tok

    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=pick_dtype(cfg.dtype), **kwargs)
    except TypeError:  # 老版本 transformers 只认 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=pick_dtype(cfg.dtype), **kwargs)

    model.config.pad_token_id = tok.pad_token_id
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tok


def load_trainable_model(name: str, cfg: ModelConfig, adapter: str | None = None):
    """加载可训练模型：基座 + LoRA。

    ``adapter`` 不为空时**接着已有的 adapter 训**（GRPO 从 SFT 的 adapter 起步），
    而不是新开一个随机初始化的 LoRA——后者会把你 SFT 学到的东西全部丢掉。

    梯度检查点必须在 **peft 包装之前**、直接开在基座 HF 模型上：
    PeftModel 只是代理 ``gradient_checkpointing_enable``，在某些 peft 版本里会被
    吞掉（表现为"设置了但显存一点没省"）。这里顺手把状态打印出来，方便排查。
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    base, tok = load_base(name, cfg)
    if cfg.grad_checkpointing:
        # 顺序很重要：先对基座开启，再包 LoRA
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        # LoRA + 梯度检查点必须打开这个，否则 adapter 拿不到梯度（经典坑）
        base.enable_input_require_grads()

    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)

    model.config.use_cache = False
    return model, tok


def grad_setup(model) -> dict:
    """训练前自检：检查点开没开、可训练参数有多少、显存占了多少。

    这三个数一眼就能看出"为什么 OOM"——之前两次 OOM 都是因为这里没打印。
    """
    inner = getattr(model, "base_model", model)
    inner = getattr(inner, "model", inner)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "grad_checkpointing": bool(getattr(inner, "is_gradient_checkpointing", False)),
        "input_require_grads": bool(
            getattr(inner, "_require_grads_hook", None) is not None
            or getattr(model, "is_prompt_learning", False)),
        "trainable_pct": round(trainable / total * 100, 3) if total else 0.0,
        "cuda_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else 0.0,
    }


def load_for_eval(base_name: str, cfg: ModelConfig, adapter: str | None = None):
    """评测用：加载基座 +（可选）LoRA adapter。"""
    model, tok = load_base(base_name, cfg)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    return model, tok


@contextlib.contextmanager
def adapter_disabled(model):
    """临时关掉 LoRA（用同一份权重算参考策略的 logp，省一个 4B 模型）。"""
    if not hasattr(model, "disable_adapter_layers"):
        yield
        return
    model.disable_adapter_layers()
    try:
        yield
    finally:
        model.enable_adapter_layers()


class _FenceStop:
    """生成到"```sql ... ```"闭合就停。

    为什么需要：Qwen3-4B-Instruct 的基座模型即使 system prompt 写了"只输出 SQL"，
    也常在代码块前后补一段解释；不截断就会一直生成到 ``max_new_tokens`` 上限。
    实测 256 token 上限下每题要 18~20 秒，而真正的 SQL 只占 ~50 token——
    **9 成时间花在生成废话上**。这里只在"代码块已闭合"时停，不改变任何有效输出。

    用纯 Python 实现 ``StoppingCriteria`` 而不是 transformers 的 LogitsProcessor：
    只需要看已生成的文本，不需要碰 logits。
    """

    def __init__(self, tokenizer, prompt_len: int, min_new_tokens: int = 6):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.min_new_tokens = min_new_tokens

    def __call__(self, input_ids, scores=None, **kwargs) -> bool:
        seq = input_ids[0, self.prompt_len:]
        if seq.shape[0] < self.min_new_tokens:
            return False
        text = self.tokenizer.decode(seq, skip_special_tokens=True)
        start = text.find("```")
        if start < 0:
            return False
        return text.find("```", start + 3) >= 0


@torch.inference_mode()
def generate_text(model, tokenizer, messages: list[dict], *,
                  max_new_tokens: int = 256, temperature: float = 0.0,
                  top_p: float = 0.95, stop_at_fence: bool = True) -> str:
    """按 chat 模板生成一轮回复（只返回新生成的文本）。

    ``stop_at_fence=True`` 时在第一个闭合代码块处停止（见 :class:`_FenceStop`），
    否则会一直生成到 ``max_new_tokens``（实测 9 成时间花在代码块外的废话上）。
    """
    ids = apply_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        # 显式打开 KV cache：训练时为了配合梯度检查点把 config.use_cache 关了，
        # 但生成时关掉会让每个 token 都重算整段前缀——慢 5 倍还多吃显存。
        "use_cache": True,
    }
    if stop_at_fence:
        gen_kwargs["stopping_criteria"] = [_FenceStop(tokenizer, len(ids))]
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(input_ids=input_ids, **gen_kwargs)
    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total_b": round(total / 1e9, 3),
        "params_trainable_m": round(trainable / 1e6, 2),
        "trainable_ratio": round(trainable / total, 5) if total else 0.0,
    }


def vram_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    }


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
