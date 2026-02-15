from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL = None
_TOKENIZER = None
_MODEL_STATIC_INFO = None


def _build_model_static_info(model, tokenizer, model_path: str) -> Dict[str, Any]:
    params = list(model.parameters())
    buffers = list(model.buffers()) if hasattr(model, "buffers") else []
    param_count = int(sum(int(p.numel()) for p in params))
    trainable_count = int(sum(int(p.numel()) for p in params if getattr(p, "requires_grad", False)))
    param_mem = int(sum(int(p.numel()) * int(p.element_size()) for p in params))
    buffer_mem = int(sum(int(b.numel()) * int(b.element_size()) for b in buffers))
    first_param = params[0] if params else None
    dtype = str(first_param.dtype).replace("torch.", "") if first_param is not None else "unknown"
    device = str(first_param.device) if first_param is not None else "cpu"
    tok_vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)

    return {
        "model_path": model_path,
        "device": device,
        "dtype": dtype,
        "parameters": param_count,
        "trainable_parameters": trainable_count,
        "tokenizer_vocab": tok_vocab,
        "param_mem_gb": float(param_mem) / (1024.0 ** 3),
        "buffer_mem_gb": float(buffer_mem) / (1024.0 ** 3),
    }


def load_ai_model(settings) -> Tuple[Any, Any]:
    global _MODEL, _TOKENIZER, _MODEL_STATIC_INFO
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    model_path = settings.local_dir if settings.local_dir else settings.model_id

    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    _MODEL.eval()
    try:
        _MODEL_STATIC_INFO = _build_model_static_info(_MODEL, _TOKENIZER, model_path)
    except Exception:
        _MODEL_STATIC_INFO = None
    return _MODEL, _TOKENIZER


def get_model_name(settings) -> str:
    # keep it short like the web UI
    if settings.local_dir:
        return settings.local_dir.replace("\\", "/")
    return settings.model_id


def get_model_runtime_stats(settings=None) -> Dict[str, Any]:
    loaded = _MODEL is not None and _TOKENIZER is not None
    model_path = ""
    if settings is not None:
        model_path = settings.local_dir if settings.local_dir else settings.model_id

    out: Dict[str, Any] = {
        "loaded": bool(loaded),
        "model_path": model_path,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": "",
        "gpu_total_gb": 0.0,
        "gpu_reserved_gb": 0.0,
        "gpu_allocated_gb": 0.0,
    }

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            out["gpu_name"] = str(props.name)
            out["gpu_total_gb"] = float(props.total_memory) / (1024.0 ** 3)
            out["gpu_reserved_gb"] = float(torch.cuda.memory_reserved()) / (1024.0 ** 3)
            out["gpu_allocated_gb"] = float(torch.cuda.memory_allocated()) / (1024.0 ** 3)
        except Exception:
            pass

    global _MODEL_STATIC_INFO
    if loaded and _MODEL_STATIC_INFO is None:
        try:
            _MODEL_STATIC_INFO = _build_model_static_info(_MODEL, _TOKENIZER, out["model_path"])
        except Exception:
            _MODEL_STATIC_INFO = None

    if _MODEL_STATIC_INFO:
        out.update(_MODEL_STATIC_INFO)

    return out


def generate_code(history: List[Dict[str, str]], system_prompt: str, settings) -> str:
    model, tokenizer = load_ai_model(settings)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # history is user/assistant messages only
    messages.extend(history)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
    else:
        # fallback
        prompt = system_prompt + "\n\n"
        for m in history:
            prompt += f"{m['role']}: {m['content']}\n"
        prompt += "assistant:"
        inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_new = min(1024, settings.max_tokens)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True if settings.temperature > 0 else False,
            temperature=settings.temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    # only decode newly generated part
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True) if hasattr(tokenizer, "decode") else ""

    return text.strip()
