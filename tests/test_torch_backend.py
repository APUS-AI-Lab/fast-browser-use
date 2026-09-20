"""Offline contracts and tiny random Qwen tests; never download pretrained weights."""

import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use import model
from fast_browser_use.torch_backend import TorchModel, load_torch_model, torch_device, torch_dtype


@pytest.mark.parametrize("system, machine, expected", [
    ("Darwin", "arm64", "mlx"), ("Darwin", "x86_64", "torch"),
    ("Linux", "x86_64", "torch"), ("Linux", "aarch64", "torch"), ("Windows", "AMD64", "torch"),
])
def test_backend_auto_selection(monkeypatch, system, machine, expected):
    monkeypatch.setenv("FBU_BACKEND", "auto")
    monkeypatch.setattr(model.platform, "system", lambda: system)
    monkeypatch.setattr(model.platform, "machine", lambda: machine)
    assert model.resolve_backend() == expected
    assert model.resolve_backend("torch") == "torch"
    assert model.resolve_backend("mlx") == "mlx"


def test_invalid_backend_does_not_load_any_model(monkeypatch):
    monkeypatch.setenv("FBU_BACKEND", "remote")
    monkeypatch.setattr(model, "_engine", None)
    with pytest.raises(ValueError, match="FBU_BACKEND"):
        model.get_model()


def test_torch_initialization_is_lazy_and_singleton(monkeypatch):
    factory = Mock(return_value=object())
    monkeypatch.setenv("FBU_BACKEND", "torch")
    monkeypatch.setattr(model, "_engine", None)
    monkeypatch.setattr("fast_browser_use.torch_backend.TorchModel", factory)
    assert model.get_model() is model.get_model()
    factory.assert_called_once_with()


def supported_config():
    return {"model_type": "qwen3_5", "text_config": {
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 12288,
    }}


@pytest.mark.parametrize("invalid", [
    {"quantization": {"bits": 4}}, {"quantization_config": {"bits": 4}},
    {"model_type": "other"}, {"text_config": {"hidden_size": 2048}},
])
def test_torch_rejects_incompatible_weights_before_loading(tmp_path, monkeypatch, invalid):
    (tmp_path / "config.json").write_text(json.dumps({**supported_config(), **invalid}))
    loader = Mock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Qwen3_5ForCausalLM=loader))
    with pytest.raises(ValueError):
        load_torch_model(tmp_path, "cpu", "float32")
    loader.from_pretrained.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_torch_loader_is_local_only_and_rejects_missing_language_weights(tmp_path, monkeypatch, missing):
    (tmp_path / "config.json").write_text(json.dumps(supported_config()))
    network_model, tokenizer = Mock(), Mock()
    loader = Mock()
    loader.from_pretrained.return_value = (
        network_model, {"missing_keys": ["model.layers.0.weight"] if missing else []},
    )
    tokenizer_loader = Mock()
    tokenizer_loader.from_pretrained.return_value = tokenizer
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3_5ForCausalLM=loader, AutoTokenizer=tokenizer_loader,
    ))
    if missing:
        with pytest.raises(ValueError, match="Incomplete"):
            load_torch_model(tmp_path, "cpu", "float32")
        tokenizer_loader.from_pretrained.assert_not_called()
    else:
        assert load_torch_model(tmp_path, "cpu", "float32") == (network_model.eval.return_value, tokenizer)
        tokenizer_loader.from_pretrained.assert_called_once_with(
            tmp_path, local_files_only=True, trust_remote_code=False,
        )
    loader.from_pretrained.assert_called_once_with(
        tmp_path, dtype="float32", device_map={"": "cpu"}, local_files_only=True,
        trust_remote_code=False, output_loading_info=True,
    )


def supported_moe_config():
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
    }


def test_torch_loader_loads_supported_moe_35b(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(supported_moe_config()))
    network_model, tokenizer = Mock(), Mock()
    moe_loader = Mock()
    moe_loader.from_pretrained.return_value = (network_model, {"missing_keys": []})
    tokenizer_loader = Mock()
    tokenizer_loader.from_pretrained.return_value = tokenizer
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3_5ForCausalLM=Mock(), Qwen3_5MoeForCausalLM=moe_loader, AutoTokenizer=tokenizer_loader,
    ))
    assert load_torch_model(tmp_path, "cpu", "float32") == (network_model.eval.return_value, tokenizer)
    moe_loader.from_pretrained.assert_called_once_with(
        tmp_path, dtype="float32", device_map={"": "cpu"}, local_files_only=True,
        trust_remote_code=False, output_loading_info=True,
    )
    tokenizer_loader.from_pretrained.assert_called_once_with(
        tmp_path, local_files_only=True, trust_remote_code=False,
    )


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 4096), ("num_hidden_layers", 32), ("moe_intermediate_size", 1024),
    ("shared_expert_intermediate_size", None), ("num_experts", 128), ("num_experts_per_tok", 4),
])
def test_torch_rejects_incompatible_moe_shapes(tmp_path, monkeypatch, field, value):
    observed = supported_moe_config()
    observed["text_config"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(observed))
    moe_loader = Mock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3_5ForCausalLM=Mock(), Qwen3_5MoeForCausalLM=moe_loader, AutoTokenizer=Mock(),
    ))
    with pytest.raises(ValueError, match="Qwen3.5-35B-A3B"):
        load_torch_model(tmp_path, "cpu", "float32")
    moe_loader.from_pretrained.assert_not_called()


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_device_auto_and_explicit_unavailable_cuda(torch, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FBU_DEVICE", "auto")
    assert str(torch_device(torch)) == "cpu"
    monkeypatch.setenv("FBU_DEVICE", "cuda")
    with pytest.raises(ValueError, match="unavailable"):
        torch_device(torch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("FBU_DEVICE", "cuda:1")
    assert str(torch_device(torch)) == "cuda:1"
    monkeypatch.setenv("FBU_DEVICE", "cuda:2")
    with pytest.raises(ValueError, match="unavailable"):
        torch_device(torch)


def test_cpu_precision_default_and_validation(torch, monkeypatch):
    monkeypatch.setenv("FBU_DTYPE", "auto")
    assert torch_dtype(torch, torch.device("cpu")) == torch.float32
    monkeypatch.setenv("FBU_DTYPE", "float16")
    with pytest.raises(ValueError, match="CPU"):
        torch_dtype(torch, torch.device("cpu"))


@pytest.fixture
def engine(torch):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1,
                         "mrope_section": [2, 3, 3]}, eos_token_id=299,
    )
    engine = TorchModel.__new__(TorchModel)
    engine.torch, engine.device = torch, torch.device("cpu")
    engine.model = Qwen3_5ForCausalLM(config).eval()
    engine.name, engine.lock = "tiny-random-offline", threading.Lock()
    engine.labels, engine.label_ids = ["A", "B", "C"], [17, 3, 99]
    engine.tokenizer = Mock()
    engine.tokenizer.apply_chat_template.return_value = "native prompt"
    engine.tokenizer.encode.return_value = [1, 2, 3, 4]
    engine.tokenizer.decode.side_effect = lambda ids, **_: "".join(chr(1000 + i) for i in ids)
    return engine


def test_real_qwen_scores_only_candidates_from_native_template(engine):
    torch = engine.torch
    with torch.inference_mode():
        expected = torch.softmax(engine.model(torch.tensor([[1, 2, 3, 4]])).logits[0, -1, [17, 3, 99]], -1)
    scores, telemetry = engine.score("observed actions", 3)
    assert scores == pytest.approx(expected.tolist())
    assert telemetry["usage"]["cached_tokens"] == 0 and not telemetry["cache_hit"]
    engine.tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "observed actions"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    engine.tokenizer.encode.assert_called_once_with("native prompt", add_special_tokens=False)
    assert engine.score("another page", 3)[0] == pytest.approx(scores)


def test_real_qwen_cached_decode_matches_fresh_forward_and_does_not_leak_inference_mode(engine):
    torch, ids = engine.torch, [1, 2, 3, 4]
    engine.model.generation_config.eos_token_id = None
    stream = engine._stream("prompt", 4)
    for response in stream:
        assert not torch.is_inference_mode_enabled()
        with torch.inference_mode():
            expected = int(engine.model(torch.tensor([ids]), use_cache=False).logits[0, -1].argmax())
        assert response.token == expected
        ids.append(expected)
    assert len(ids) == 8


def test_transformers_maps_multimodal_checkpoint_to_text_weights(engine, tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    # Tiny random tensors use the official checkpoint's key layout. No Hub access.
    config = {"model_type": "qwen3_5", "text_config": engine.model.config.to_dict()}
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = {key.replace("model.", "model.language_model.", 1): value.contiguous()
               for key, value in engine.model.state_dict().items()}
    save_file(weights, tmp_path / "model.safetensors")
    loaded, info = Qwen3_5ForCausalLM.from_pretrained(
        tmp_path, local_files_only=True, trust_remote_code=False, output_loading_info=True,
        device_map={"": "cpu"}, dtype=engine.torch.float32,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    for name, value in engine.model.state_dict().items():
        assert engine.torch.equal(loaded.state_dict()[name], value)


def test_field_generation_waits_for_complete_unicode_and_stops_at_valid_json(engine):
    torch = engine.torch
    tokens = iter([10, 11, 12])

    def forward(**_kwargs):
        logits = torch.zeros(1, 1, 300)
        logits[0, 0, next(tokens)] = 10
        return SimpleNamespace(logits=logits, past_key_values=object())

    engine.model = Mock(side_effect=forward, generation_config=SimpleNamespace(eos_token_id=299))
    engine.tokenizer.decode.side_effect = lambda ids, **_: '"Z\ufffd' if len(ids) == 1 else '"Zürich"}'
    value, telemetry = engine.generate_text({"goal": "Enter Zürich", "field": {"label": "City"}})
    assert value == "Zürich" and telemetry["usage"]["completion_tokens"] == 2
    assert engine.model.call_count == 2
    assert engine.tokenizer.encode.call_args.args[0] == 'native prompt{"text":'


def test_early_eos_never_types_partial_json(engine):
    torch = engine.torch
    logits = torch.zeros(1, 1, 300)
    logits[0, 0, 299] = 1
    engine.model = Mock(return_value=SimpleNamespace(logits=logits, past_key_values=None),
                        generation_config=SimpleNamespace(eos_token_id=[298, 299]))
    with pytest.raises(ValueError, match="nothing typed"):
        engine.generate_text({"goal": "Enter the city"})
    engine.model.assert_called_once()
