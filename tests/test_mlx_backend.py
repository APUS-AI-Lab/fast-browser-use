"""Tiny random MLX models: chunking and prefix reuse must preserve candidate scores."""

import contextlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use import model


@pytest.fixture(params=[0, 4], ids=["dense", "moe"])
def engine(request):
    mx = pytest.importorskip("mlx.core")
    qwen = pytest.importorskip("mlx_lm.models.qwen3_5")
    mx.random.seed(17)
    args = qwen.ModelArgs(model_type="qwen3_5", text_config={
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 64,
        "vocab_size": 256, "full_attention_interval": 2,
        "linear_num_value_heads": 2, "linear_num_key_heads": 1,
        "linear_key_head_dim": 64, "linear_value_head_dim": 64,
        "num_experts": request.param, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
    })
    result = model.LocalModel.__new__(model.LocalModel)
    result.mx, result.model = mx, qwen.Model(args)
    result.model.eval()
    result.lock, result.name, result.prefixes = threading.Lock(), "tiny-random-qwen", {}
    result.label_ids = [65, 66, 67]
    result.tokenizer = SimpleNamespace(
        encode=lambda value, **_: list(value.encode()),
        apply_chat_template=lambda messages, **_: "User: " + messages[0]["content"] + "\nAssistant:",
    )
    return result


@pytest.mark.parametrize("chunk_size", [7, 2048])
def test_chunked_candidate_scores_and_cached_prefix_match_full_forward(engine, monkeypatch, chunk_size):
    from mlx_lm.models.cache import make_prompt_cache

    mx = engine.mx
    monkeypatch.setattr(model, "MLX_PREFILL_STEP_SIZE", chunk_size)
    prompts = [
        "Choose an observed action.\nGOAL: Search\nPAGE:\nSearch field is empty.",
        "Choose an observed action.\nGOAL: Search\nPAGE:\nSearch field contains a new value.",
        "Choose an observed action.\nGOAL: Save\nPAGE:\nSave button is visible.",
    ]
    for content, expected_hit in zip(prompts, [False, True, False], strict=True):
        tokens = engine.tokenizer.encode(engine.template(content))
        logits = engine.model(mx.array([tokens]), cache=make_prompt_cache(engine.model))[0, -1]
        expected = mx.softmax(logits[mx.array(engine.label_ids)].astype(mx.float32))
        mx.eval(expected)
        scores, telemetry = engine.score(content, 3)
        assert scores == pytest.approx(expected.tolist(), abs=1e-4)
        assert telemetry["cache_hit"] is expected_hit
        assert telemetry["prefill_step_size"] == chunk_size
        assert telemetry["usage"]["prompt_tokens"] == len(tokens)


def test_scoring_error_exits_residency_context_and_releases_lock(monkeypatch):
    events = []

    @contextlib.contextmanager
    def residency(_model):
        events.append("enter")
        try:
            yield
        finally:
            events.append("restore")

    monkeypatch.setitem(sys.modules, "mlx_lm.generate", SimpleNamespace(wired_limit=residency))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", SimpleNamespace(make_prompt_cache=Mock()))
    engine = model.LocalModel.__new__(model.LocalModel)
    engine.lock, engine.model = threading.Lock(), object()
    engine.template = Mock(side_effect=ValueError("Invalid template"))
    with pytest.raises(ValueError, match="Invalid template"):
        engine.score("Choose an observed action", 2)
    assert events == ["enter", "restore"]
    assert not engine.lock.locked()
