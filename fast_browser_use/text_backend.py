"""Load supported dense and MoE Qwen3.5 MLX 4-bit models through the upstream text backend."""

import json
from pathlib import Path


def load_text_model(location):
    config = json.loads((Path(location) / "config.json").read_text())
    text = config.get("text_config", {})
    quantization = config.get("quantization", {})
    dense_9b = (
        config.get("model_type") == "qwen3_5" and text.get("hidden_size") == 4096
        and text.get("num_hidden_layers") == 32 and text.get("intermediate_size") == 12288
    )
    moe_35b = (
        config.get("model_type") == "qwen3_5_moe" and text.get("hidden_size") == 2048
        and text.get("num_hidden_layers") == 40 and text.get("moe_intermediate_size") == 512
        and text.get("shared_expert_intermediate_size") == 512
        and text.get("num_experts") == 256 and text.get("num_experts_per_tok") == 8
    )
    if not (dense_9b or moe_35b) or quantization.get("bits") != 4:
        raise ValueError("This release supports Qwen3.5-9B and Qwen3.5-35B-A3B MLX 4-bit weights only")
    from mlx_lm import load

    # Upstream discards the vision weights and strictly loads the language model.
    return load(location)
