"""Independent assertions must fail on premature DONE and partial outcomes."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use.verification import verify_outcome


def test_outcome_requires_evidence():
    browser = SimpleNamespace(evaluate=Mock())
    with pytest.raises(ValueError, match="at least one"):
        verify_outcome(browser)
    browser.evaluate.assert_not_called()


@pytest.mark.parametrize("expected", [{"text": "Saved"}, {"text": [""]}, {"title": " "}, {"url": True}])
def test_empty_or_malformed_assertions_cannot_pass(expected):
    browser = SimpleNamespace(evaluate=Mock())
    with pytest.raises(ValueError):
        verify_outcome(browser, **expected)
    browser.evaluate.assert_not_called()


def test_outcome_reads_fresh_facts_and_requires_every_constraint():
    browser = SimpleNamespace(evaluate=Mock(return_value={
        "url": "https://example.test/results", "title": "Results", "text": "Saved Paris Economy",
    }))
    expected = {"url": "https://example.test/results", "title": "Results", "text": ["Saved", "Paris"]}
    assert verify_outcome(browser, **expected)["passed"]
    assert not verify_outcome(browser, **{**expected, "url": "https://example.test/results#done"})["passed"]
    result = verify_outcome(browser, **{**expected, "text": ["Paris", "Business"]})
    assert not result["passed"]
    assert result["checks"]["text_1"] and not result["checks"]["text_2"]
    assert result["expected"]["text"] == ["Paris", "Business"]
