"""No weights/network: test the boundary between model preferences and executable actions."""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use import model


def state():
    return {
        "url": "https://example.test",
        "title": "Search",
        "text": "City",
        "actions": [
            {"id": "e1", "kind": "fill", "node": 1, "role": "textbox", "label": "City", "value": ""},
            {"id": "e2", "kind": "click", "node": 2, "role": "button", "label": "Search"},
            {"id": "wait", "kind": "wait", "label": "Wait for loading"},
        ],
    }


def test_code_compiler_rejects_collisions_and_multitoken_labels():
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda s, **_: [7] if s in {"A", "B"} else [8] if s == "C" else [1, 2]
    assert model.candidate_codes(tokenizer, 2) == (["A", "C"], [7, 8])
    with pytest.raises(ValueError, match="unique action codes"):
        model.candidate_codes(tokenizer, 3)


def test_joint_choice_cannot_mix_operation_and_target(monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "1")
    engine = Mock(labels=list("ABCDE"))
    engine.score.side_effect = [
        ([0.9, 0.1], {"model": "local", "latency_ms": 5, "usage": {}}),
        ([0.03, 0.90, 0.05, 0.02], {"model": "local", "latency_ms": 10, "usage": {}}),
    ]
    monkeypatch.setattr(model, "get_model", lambda: engine)
    result = model.choose(state(), "Search for Lisbon", [])
    assert (result["choice"], result["operation"], result["target"]) == ("e2", "CLICK", "2")
    assert engine.score.call_count == 2
    assert "Search for Lisbon" in engine.score.call_args.args[0]
    assert result["confidence"] == 0.9


@pytest.mark.parametrize("scores", [[1], [float("nan"), 0, 0, 0, 0], [-1, 2, 0, 0, 0], [0.1] * 5])
def test_invalid_scores_never_become_an_action(scores):
    with pytest.raises(ValueError):
        model.decision_from_scores(model.legal_candidates(state()), scores)


def test_low_score_is_not_artificially_raised():
    result = model.decision_from_scores(model.legal_candidates(state()), [0.2] * 5)
    assert result["confidence"] == 0.2


@pytest.mark.parametrize("selected", ["e1", "e2", "wait", "DONE", "BLOCKED"])
def test_full_goal_scores_completion_and_observed_actions_once(monkeypatch, selected):
    monkeypatch.delenv("FBU_PLAN", raising=False)
    monkeypatch.delenv("FBU_DECISION_MODE", raising=False)
    candidates = model.legal_candidates(state())
    scores = [float(c["id"] == selected) for c in candidates]
    engine = Mock(labels=list("ABCDE"))
    engine.score.return_value = (scores, {"model": "local", "latency_ms": 5, "usage": {}})
    monkeypatch.setattr(model, "get_model", lambda: engine)
    result = model.choose(state(), "Find articles about Lisbon", [])
    assert result["choice"] == selected
    assert result["scoring_passes"] == 1 and result["completion_check"] is None
    engine.score.assert_called_once()
    prompt = engine.score.call_args.args[0]
    for candidate in candidates:
        assert candidate["description"] in prompt
    assert state()["text"] in prompt and state()["url"] in prompt
    assert 'City contains ""' in prompt


def test_unknown_decision_mode_never_calls_model(monkeypatch):
    monkeypatch.setenv("FBU_DECISION_MODE", "guess")
    engine = Mock()
    monkeypatch.setattr(model, "get_model", lambda: engine)
    with pytest.raises(ValueError, match="FBU_DECISION_MODE"):
        model.choose(state(), "Search", [])
    engine.score.assert_not_called()


@pytest.mark.parametrize(
    "text",
    ['{"text":null}', '{"text":1}', '{"text":"x","code":"y"}', 'Thoughts: {"text":"Lisbon"}', '{"text":""}', "[]"],
)
def test_field_text_must_be_exact_json(text):
    with pytest.raises(ValueError):
        model.parse_field_text(text)


def test_field_text_preserves_quotes_and_unicode():
    assert model.parse_field_text('{"text":"Zürich"}') == "Zürich"


def test_field_recovery_retains_values_from_the_full_user_request():
    context = model.field_context(
        "Submit the search", {"label": "City"}, state(), [], task="Find articles about Lisbon"
    )
    assert context["goal"] == "Find articles about Lisbon"
    assert context["subgoal"] == "Submit the search"
    assert context["field"]["label"] == "City"


def test_native_select_values_are_observed_not_generated():
    p = state()
    p["actions"].insert(
        0,
        {
            "id": "s1",
            "kind": "select",
            "node": 9,
            "label": "Category → Design",
            "value": "design",
            "current_value": "All",
            "role": "combobox",
        },
    )
    candidates = model.legal_candidates(p)
    c = next(c for c in candidates if c["id"] == "s1")
    assert c["operation"] == "SELECT" and c["target"] == "1:1"
    assert "Category → Design" in c["description"]
    assert p["actions"][0]["value"] == "design"


def test_completion_is_a_model_decision_not_an_automatic_step(monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "1")
    engine = Mock(labels=list("ABCDE"))
    engine.score.return_value = ([0.07, 0.93], {"model": "local", "latency_ms": 5, "usage": {}})
    monkeypatch.setattr(model, "get_model", lambda: engine)
    result = model.choose(state(), "Enter the city", [])
    assert result["choice"] == "DONE"
    assert result["confidence"] == 0.93
    assert engine.score.call_count == 1


def test_completion_tie_continues_without_exposing_pseudo_action(monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "1")
    engine = Mock(labels=list("ABCDE"))
    engine.score.side_effect = [
        ([0.5, 0.5], {"model": "local", "latency_ms": 5, "usage": {}}),
        ([0.1, 0.8, 0.05, 0.05], {"model": "local", "latency_ms": 5, "usage": {}}),
    ]
    monkeypatch.setattr(model, "get_model", lambda: engine)
    assert model.choose(state(), "Click Search", [])["choice"] == "e2"


def test_completion_observes_settings_shown_in_button_labels(monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "1")
    page = state()
    page["actions"][1]["label"] = "Select dates Mon, Sep 21 — Tue, Sep 22"
    engine = Mock(labels=list("ABCDE"))
    engine.score.return_value = ([0.1, 0.9], {"model": "local", "latency_ms": 5, "usage": {}})
    monkeypatch.setattr(model, "get_model", lambda: engine)
    model.choose(page, "Set the end date to 22 September", [])
    assert "Tue, Sep 22" in engine.score.call_args.args[0]


def test_completion_sees_task_results_and_untouched_filter_values(monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "0")
    monkeypatch.setenv("FBU_DECISION_MODE", "binary")
    page = state()
    page["text"] = "Search results for the requested dates"
    page["actions"].append(
        {"id": "e3", "kind": "click", "node": 3, "role": "checkbox", "label": "Has figures", "checked": "false"}
    )
    engine = Mock(labels=list("ABCDE"))
    engine.score.return_value = ([0.1, 0.9], {"model": "local", "latency_ms": 5, "usage": {}})
    monkeypatch.setattr(model, "get_model", lambda: engine)
    model.choose(page, "Submit the search", [{"kind": "fill", "action": "City"}], task="Find articles with figures")
    prompt = engine.score.call_args.args[0]
    assert "Find articles with figures" in prompt
    assert page["text"] in prompt
    assert "Has figures is unchecked" in prompt


def test_checklist_completion_keeps_future_task_constraints_out(monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "1")
    engine = Mock(labels=list("ABCDE"))
    engine.score.return_value = ([0.1, 0.9], {"model": "local", "latency_ms": 5, "usage": {}})
    monkeypatch.setattr(model, "get_model", lambda: engine)
    model.choose(state(), "Enter Lisbon", [], task="Enter Lisbon, then filter and submit the form")
    prompt = engine.score.call_args.args[0]
    assert "SUBGOAL: Enter Lisbon" in prompt
    assert "then filter and submit" not in prompt
    assert "USER TASK:" not in prompt


def test_local_deliberation_cannot_create_an_executable_action(monkeypatch):
    engine = Mock(labels=list("ABCDE"))
    explanation = 'Use document.querySelector("#invented").click()'
    engine.think_score.return_value = (
        [0.9, 0.1],
        {"model": "local", "latency_ms": 10, "usage": {}, "reasoning": {"latency_ms": 8, "text": explanation}},
    )
    engine.score.return_value = ([0.03, 0.9, 0.04, 0.03], {"model": "local", "latency_ms": 5, "usage": {}})
    monkeypatch.setenv("FBU_REASONING", "1")
    monkeypatch.setattr(model, "get_model", lambda: engine)
    result = model.choose(state(), "Click Search", [], task="Find articles")
    assert result["choice"] == "e2"
    assert result["operation"] == "CLICK"
    assert result["reasoning"]["text"] == explanation
    assert "Click Search" in engine.think_score.call_args.args[0]


@pytest.mark.parametrize("finished", [False, True])
def test_thought_channel_must_close_before_scoring_an_answer_slot(monkeypatch, finished):
    response = SimpleNamespace(token=99 if finished else 1, prompt_tokens=10, generation_tokens=1)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=lambda *_a, **_k: iter([response])))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=lambda **_k: None))
    engine = model.LocalModel.__new__(model.LocalModel)
    engine.lock, engine.model, engine.name = threading.Lock(), object(), "local"
    engine.tokenizer = Mock()
    engine.tokenizer.encode.return_value = [99]
    engine.tokenizer.decode.return_value = "I will scroll.</think>"
    engine.template = Mock(return_value="<|im_start|>assistant\n<think>\n")
    engine.score = Mock(return_value=([0.2, 0.8], {"latency_ms": 1}))
    if not finished:
        with pytest.raises(ValueError, match="budget exhausted"):
            engine.think_score("Choose from observed actions", 2)
        engine.score.assert_not_called()
    else:
        scores, _ = engine.think_score("Choose from observed actions", 2)
        assert scores == [0.2, 0.8]
        assert engine.score.call_args.kwargs["continuation"].endswith("</think>\nAction code: ")


def test_qwen_thinking_uses_the_boundary_from_its_chat_template(monkeypatch):
    response = SimpleNamespace(token=99, prompt_tokens=10, generation_tokens=1)
    stream = Mock(return_value=iter([response]))
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=lambda **_k: None))
    engine = model.LocalModel.__new__(model.LocalModel)
    engine.lock, engine.model, engine.name = threading.Lock(), object(), "local"
    engine.tokenizer = Mock()
    engine.tokenizer.encode.return_value = [99]
    engine.tokenizer.decode.return_value = "Already submitted.</think>"
    engine.template = Mock(return_value="<|im_start|>assistant\n<think>\n")
    engine.score = Mock(return_value=([0.2, 0.8], {"latency_ms": 1}))
    _, telemetry = engine.think_score("Check completion", 2)
    engine.tokenizer.encode.assert_called_once_with("</think>", add_special_tokens=False)
    assert stream.call_args.kwargs["prompt"].endswith("<think>\n")
    assert engine.score.call_args.kwargs["continuation"] == "Already submitted.</think>\nAction code: "
    assert telemetry["reasoning"]["end_marker"] == "</think>"
