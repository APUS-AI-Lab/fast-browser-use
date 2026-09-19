"""Independent outcome checks, never inputs to the action policy."""

import base64
from urllib.parse import parse_qs, urlparse


def validate_expectations(*, url=None, title=None, text=()):
    if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in (url, title)):
        raise ValueError("Expected URL/title must be nonempty strings")
    if not isinstance(text, (list, tuple)) or any(not isinstance(t, str) or not t.strip() for t in text):
        raise ValueError("Expected text must be a list of nonempty strings")
    if not url and not title and not text:
        raise ValueError("Supply at least one expected URL, title or visible text fragment")


def verify_outcome(browser, *, url=None, title=None, text=()):
    """Check caller-supplied end conditions against a fresh read, never the model's claim.

    URL/title are exact matches; every text fragment must occur in rendered body text.
    These assertions are not passed to the planner, action selector or field generator.
    """
    validate_expectations(url=url, title=title, text=text)
    facts = browser.evaluate("({url:location.href,title:document.title,text:document.body.innerText})")
    checks = {}
    if url:
        checks["url"] = facts["url"] == url
    if title:
        checks["title"] = facts["title"] == title
    for index, fragment in enumerate(text):
        checks[f"text_{index + 1}"] = fragment in facts["text"]
    return {
        "passed": all(checks.values()), "checks": checks,
        "expected": {"url": url, "title": title, "text": list(text)},
        "observed": {"url": facts["url"], "title": facts["title"]},
    }


def verify_flights(page):
    """Independent checks on the resulting page, not the model's DONE answer."""
    parsed = urlparse(page["url"])
    encoded = parse_qs(parsed.query).get("tfs", [""])[0]
    try:
        date_in_url = b"2026-09-20" in base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except ValueError:
        date_in_url = False
    actions = page["actions"]
    values = {a["label"].strip(): a.get("value") for a in actions}
    flights = [a["label"] for a in actions if "Select flight" in a["label"]]
    checks = {
        "search_page": parsed.hostname == "www.google.com" and parsed.path == "/travel/flights/search",
        "one_way": values.get("Change ticket type. One way") == "One way",
        "origin": values.get("Where from?") == "Zürich",
        "destination": values.get("Where to?") == "London",
        "date": values.get("Departure") == "Sun, Sep 20",
        "year": date_in_url or "departing 2026-09-20" in page["text"],
        "results": bool(flights) and all("Sunday, September 20" in f for f in flights),
    }
    return {"passed": all(checks.values()), "checks": checks, "visible_flights": flights}
