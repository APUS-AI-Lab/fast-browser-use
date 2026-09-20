"""Local PyTorch text inference; no model-generated code or remote inference."""

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .model import LocalModel, candidate_codes, model_location


def torch_device(torch):
    requested = os.environ.get("FBU_DEVICE", "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("FBU_DEVICE must be auto, cpu, cuda or cuda:N; use MLX on Apple Silicon")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Requested CUDA device is unavailable: {requested}")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device is unavailable: {requested}")
        device = torch.device(f"cuda:{index}")
    return device


def torch_dtype(torch, device):
    requested = os.environ.get("FBU_DTYPE", "auto")
    if requested == "auto":
        if device.type == "cpu":
            requested = "float32"
        else:
            with torch.cuda.device(device):
                requested = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if requested not in {"float32", "float16", "bfloat16"}:
        raise ValueError("FBU_DTYPE must be auto, float32, float16 or bfloat16")
    if device.type == "cpu" and requested == "float16":
        raise ValueError("Use float32 or bfloat16 on CPU")
    return getattr(torch, requested)


def load_torch_model(location, device, dtype):
    config = json.loads((Path(location) / "config.json").read_text())
    text = config.get("text_config", config)
    if (config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}
            or text.get("hidden_size") != 4096 or text.get("num_hidden_layers") != 32
            or text.get("intermediate_size") != 12288):
        raise ValueError("The PyTorch backend supports Qwen3.5-9B weights only")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("PyTorch requires original Qwen3.5-9B weights, not MLX/quantized weights")
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    # Transformers extracts the text config and maps model.language_model.* keys.
    # Vision/MTP weights are unused; missing language weights must never be randomized.
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        location, dtype=dtype, device_map={"": str(device)}, local_files_only=True,
        trust_remote_code=False, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete or incompatible Qwen3.5-9B language weights")
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, trust_remote_code=False)
    return model.eval(), tokenizer


class TorchModel(LocalModel):
    backend = "torch"

    def __init__(self, path=None):
        try:
            import torch
            from transformers import Qwen3_5ForCausalLM  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install PyTorch support: uv sync --extra torch (or pip install '.[torch]')") from error
        self.torch = torch
        self.device = torch_device(torch)
        self.dtype = torch_dtype(torch, self.device)
        started = time.perf_counter()
        self.name, self.revision, self.location = model_location(path, self.backend)
        self.model, self.tokenizer = load_torch_model(self.location, self.device, self.dtype)
        self.lock = threading.Lock()
        self.labels, self.label_ids = candidate_codes(self.tokenizer, 256)
        self.load_ms = round((time.perf_counter() - started) * 1000)

    def score(self, content, count, *, purpose="action", continuation="", thinking=False):
        if not 1 <= count <= len(self.label_ids):
            raise ValueError("Invalid candidate count")
        with self.lock, self.torch.inference_mode():
            started = time.perf_counter()
            prompt = self.template(content, thinking=thinking) + continuation
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            inputs = self.torch.tensor([tokens], device=self.device)
            # Only project the final hidden state to the vocabulary. Each decision
            # sees a fresh prompt; no mutable hybrid attention cache crosses actions.
            logits = self.model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
            scores = self.torch.softmax(logits[self.label_ids[:count]].float(), dim=-1).cpu().tolist()
            return scores, {
                "model": self.name, "backend": self.backend, "device": str(self.device),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": {"prompt_tokens": len(tokens), "cached_tokens": 0, "completion_tokens": 1},
                "cache_hit": False, "candidate_count": count,
            }

    def _stream(self, prompt, max_tokens):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        inputs = self.torch.tensor([tokens], device=self.device)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        cache, generated, emitted = None, [], ""
        for index in range(max_tokens):
            # Keep inference_mode inside each step so it cannot leak across yield.
            with self.torch.inference_mode():
                output = self.model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
                token = int(output.logits[0, -1].argmax().item())
                cache = output.past_key_values
                inputs = self.torch.tensor([[token]], device=self.device)
            if token in eos:
                return
            generated.append(token)
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # A byte-level tokenizer may need several tokens to finish a Unicode character.
            text = "" if decoded.endswith("\ufffd") else decoded[len(emitted):]
            if text:
                emitted = decoded
            yield SimpleNamespace(text=text, token=token, prompt_tokens=len(tokens), generation_tokens=index + 1)
