"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from fast_browser_use import agent as loop
from fast_browser_use.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p), prepare=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "plan": ["Find a book"],
        "plan_index": 0,
        "planned": True,
        "plan_calls": [],
        "milestones": [],
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["title"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    actual = browser_operation({"operation": "observe", "session": Mock(send=cdp), "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import fast_browser_use.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation(
            {
                "operation": "act",
                "session": Mock(send=cdp),
                "action": {
                    "id": "e1",
                    "kind": "select",
                    "node": 1,
                    "value": "Design",
                },
            }
        )
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()
    assert runner.state["rejections"][0]["decision_index"] is None
    assert runner.state["rejections"][0]["execution_recorded"] is False


def test_post_action_read_failure_records_execution_without_replaying(runner, monkeypatch):
    monkeypatch.setattr(loop, "choose", Mock(return_value=decision("e3")))
    runner.state["browser"].observe.side_effect = [StalePage("navigation after input"), page()]
    runner.command("tick")
    runner.state["browser"].act.assert_called_once()
    assert len(runner.state["history"]) == 1
    rejection = runner.state["rejections"][0]
    assert rejection["decision_index"] == 0
    assert rejection["execution_recorded"] is True
    assert runner.state["decision"] is None


def test_done_advances_one_model_generated_subgoal(runner):
    runner.state["plan"] = ["Fill the query", "Submit the query"]
    runner.state["decision"] = decision("DONE")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "ready"
    assert runner.state["plan_index"] == 1
    assert runner.state["milestones"] == ["Fill the query"]
    runner.state["browser"].act.assert_not_called()
    runner.state["decision"] = decision("DONE")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "done"
    assert runner.state["plan_index"] == 2


def test_default_mode_does_not_generate_or_install_a_task_plan(runner, monkeypatch):
    monkeypatch.delenv("FBU_PLAN", raising=False)
    runner.state["planned"] = False
    planner = Mock()
    chooser = Mock(return_value=decision("e3"))
    monkeypatch.setattr(loop, "make_plan", planner)
    monkeypatch.setattr(loop, "choose", chooser)
    runner.command("predict")
    planner.assert_not_called()
    assert chooser.call_args.args[1] == runner.state["goal"]
    assert runner.state["plan"] == [runner.state["goal"]]
    assert runner.state["plan_calls"] == []


def test_full_goal_preparation_is_inside_task_timer_and_before_inference(runner, monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "0")
    runner.state["started_at"] = None
    calls = []

    def prepare(observed, **_kwargs):
        assert runner.state["started_at"] is not None
        calls.append("prepare")
        return {**observed, "preparation": {"latency_ms": 0, "samples": 2, "settled": True}}

    def choose(*_args, **_kwargs):
        calls.append("choose")
        return decision("e3")

    runner.state["browser"].prepare.side_effect = prepare
    monkeypatch.setattr(loop, "choose", choose)
    runner.command("predict")
    assert calls == ["prepare", "choose"]
    assert runner.state["observations"][0]["settled"]


def test_checklist_keeps_existing_observation_waits_by_default(runner, monkeypatch):
    monkeypatch.setenv("FBU_PLAN", "1")
    monkeypatch.delenv("FBU_SETTLE_MS", raising=False)
    monkeypatch.setattr(loop, "choose", Mock(return_value=decision("e3")))
    runner.command("predict")
    runner.state["browser"].prepare.assert_not_called()
