from unittest.mock import Mock

import pytest

from fast_browser_use import demo


@pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "", "data:text/html,test"])
def test_custom_url_must_be_http(monkeypatch, url):
    constructor = Mock()
    monkeypatch.setattr(demo, "Agent", constructor)
    with pytest.raises(ValueError, match="URL"):
        demo.Inspector("http://127.0.0.1:8767").command("reset", {"scenario": "custom", "url": url, "goal": "Read"})
    constructor.assert_not_called()


def test_custom_goal_passes_through_without_a_site_plan(monkeypatch):
    constructor = Mock()
    constructor.return_value.snapshot.return_value = {"status": "ready"}
    monkeypatch.setattr(demo, "Agent", constructor)
    inspector = demo.Inspector("http://127.0.0.1:8767")
    inspector.command("reset", {"scenario": "custom", "url": "https://example.test", "goal": "Open the contact page"})
    constructor.assert_called_once_with("https://example.test", "Open the contact page", screenshots=True)
