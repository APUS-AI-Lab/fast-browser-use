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


def test_qwen_9b_uses_upstream_text_loader(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(config()))
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
