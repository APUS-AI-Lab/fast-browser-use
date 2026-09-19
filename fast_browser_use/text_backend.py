"""Load the supported Qwen3.5-9B MLX 4-bit model through the upstream text backend."""

import json
from pathlib import Path


def load_text_model(location):
    config = json.loads((Path(location) / "config.json").read_text())
    text = config.get("text_config", {})
    quantization = config.get("quantization", {})
    if (config.get("model_type") != "qwen3_5" or text.get("hidden_size") != 4096
            or text.get("num_hidden_layers") != 32 or text.get("intermediate_size") != 12288
            or quantization.get("bits") != 4):
        raise ValueError("This release supports Qwen3.5-9B MLX 4-bit weights only")
    from mlx_lm import load

    # Upstream discards the vision weights and strictly loads the language model.
    return load(location)
