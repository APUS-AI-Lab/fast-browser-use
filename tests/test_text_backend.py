"""Supported model validation without MLX, weights, or network access."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use.text_backend import load_text_model


def config():
    return {
        "model_type": "qwen3_5", "quantization": {"bits": 4},
        "text_config": {"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 12288},
    }


def moe_config():
    return {
        "model_type": "qwen3_5_moe", "quantization": {"bits": 4},
        "text_config": {
            "hidden_size": 2048, "num_hidden_layers": 40, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512, "num_experts": 256, "num_experts_per_tok": 8,
        },
    }


@pytest.mark.parametrize("model_config", [config(), moe_config()])
def test_supported_qwen_uses_upstream_text_loader(tmp_path, monkeypatch, model_config):
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    load = Mock(return_value=("model", "tokenizer"))
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    assert load_text_model(str(tmp_path)) == ("model", "tokenizer")
    load.assert_called_once_with(str(tmp_path))


@pytest.mark.parametrize("override", [
    {"model_type": "other_architecture"}, {"text_config": {"hidden_size": 2048}},
    {"quantization": {"bits": 8}}, {},
])
def test_unsupported_or_incomplete_model_fails_before_weight_loading(tmp_path, monkeypatch, override):
    observed = {**config(), **override} if override else {}
    (tmp_path / "config.json").write_text(json.dumps(observed))
    load = Mock()
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    with pytest.raises(ValueError, match="Qwen3.5-9B"):
        load_text_model(str(tmp_path))
    load.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 4096), ("num_hidden_layers", 32), ("moe_intermediate_size", 1024),
    ("shared_expert_intermediate_size", None), ("num_experts", 128), ("num_experts_per_tok", 4),
])
def test_other_moe_shapes_fail_before_loading(tmp_path, monkeypatch, field, value):
    observed = moe_config()
    observed["text_config"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(observed))
    load = Mock()
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    with pytest.raises(ValueError, match="Qwen3.5-35B-A3B"):
        load_text_model(str(tmp_path))
    load.assert_not_called()
