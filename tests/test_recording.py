"""Headless recording retains real wall-clock waits and independent outcome checks."""

import json
import shutil
import subprocess
import time
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from playwright.sync_api import sync_playwright

from fast_browser_use import recording
from fast_browser_use.model import decision_from_scores, legal_candidates


def test_recording_metadata_does_not_require_mlx(monkeypatch):
    def version(name):
        if name.startswith("mlx"):
            raise PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(recording, "version", version)
    assert recording.runtime_versions() == {
        name: "test-version" for name in ("torch", "accelerate", "transformers", "playwright")
    }


@pytest.mark.parametrize("correct", [True, False])
def test_headless_video_preserves_inference_wait_and_rejects_false_done(tmp_path, monkeypatch, correct):
    if not shutil.which("ffprobe"):
        pytest.skip("Install ffmpeg for original-timing video verification")
    with sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip("Run fbu install-browser for headless recording checks")
    monkeypatch.setenv("FBU_HEADLESS", "0")  # Recording must still work without any display.
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("FBU_PLAN", "0")
    monkeypatch.setenv("FBU_SETTLE_MS", "0")
    monkeypatch.setattr(recording, "get_model", lambda: SimpleNamespace(
        name="test/local", backend="torch", device="cpu", dtype="float32", revision=None, load_ms=0,
    ))

    def choose(page, _goal, history, **_kwargs):
        # Block Python just as local inference does. Video must preserve this time.
        time.sleep(0.65)
        candidates = legal_candidates(page)
        selected = "DONE" if history else next(a["id"] for a in page["actions"] if a["label"] == "Finish")
        return {**decision_from_scores(candidates, [float(c["id"] == selected) for c in candidates]),
                "latency_ms": 650, "usage": {}}

    monkeypatch.setattr("fast_browser_use.agent.choose", choose)
    html = """<title>Ready</title>
    <button onclick="document.title='Finished';this.textContent='Saved'">Finish</button>"""
    folder = tmp_path / "recording"
    arguments = dict(url="data:text/html," + quote(html), goal="Finish", expected={
        "title": "Finished" if correct else "Wrong outcome",
    })
    if correct:
        recording.record(folder, **arguments)
    else:
        with pytest.raises(RuntimeError, match="failed verification"):
            recording.record(folder, **arguments)
    trace = json.loads((folder / "trace.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    assert trace["browser_headless"] is True
    assert trace["inference"] == {"backend": "torch", "device": "cpu", "dtype": "float32"}
    assert trace["verification"]["passed"] is correct
    assert len(trace["history"]) == 1 and trace["elapsed_ms"] >= 1300
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
        str(folder / "browser.webm"),
    ], text=True))
    assert duration >= summary["task_ms"] / 1000 + 0.8  # Includes the explicit 1s final hold.
    assert (folder / "result.png").stat().st_size > 0
    assert (folder / "result.html").stat().st_size > 0
