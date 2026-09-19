"""Observation retries are read-only, including during live-site redirects."""

import itertools
from unittest.mock import Mock

import pytest

from fast_browser_use import browser


def test_empty_document_and_redirect_are_observed_again_without_input(monkeypatch):
    instance = browser.Browser.__new__(browser.Browser)
    instance.session = Mock()
    empty = {"text": "", "actions": [{"kind": "wait"}]}
    ready = {"text": "Search", "actions": [{"kind": "fill", "node": 1}]}
    operation = Mock(side_effect=[empty, browser.StalePage("redirect"), ready])
    monkeypatch.setattr(browser, "browser_operation", operation)
    monkeypatch.setattr(browser.time, "sleep", Mock())
    assert instance.observe(screenshot=False) is ready
    assert operation.call_count == 3
    assert all(call.args[0]["operation"] == "observe" for call in operation.call_args_list)
    instance.session.send.assert_not_called()


def test_preparation_defers_inference_when_every_read_is_interrupted(monkeypatch):
    instance = browser.Browser.__new__(browser.Browser)
    instance.session = Mock()
    operation = Mock(side_effect=browser.StalePage("navigation"))
    monkeypatch.setattr(browser, "browser_operation", operation)
    monkeypatch.setattr(browser.time, "sleep", Mock())
    monkeypatch.setattr(browser.time, "perf_counter", lambda: next(clock))
    monkeypatch.setenv("FBU_SETTLE_MS", "150")
    clock = itertools.count(step=0.5)
    with pytest.raises(browser.StalePage, match="before inference"):
        instance.prepare({"page_key": [1], "marker": ["old"]})
    assert operation.call_count > 0
    assert all(call.args[0]["operation"] == "observe" for call in operation.call_args_list)
    instance.session.send.assert_not_called()
