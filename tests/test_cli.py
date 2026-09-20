"""CLI setup and downloads must not start browser tasks or model inference."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use import cli


@pytest.mark.parametrize("returncode", [0, 7])
def test_install_browser_uses_own_environment_and_preserves_failure(monkeypatch, returncode):
    install = Mock(return_value=SimpleNamespace(returncode=returncode))
    model = Mock(side_effect=AssertionError("Browser installation must not load model weights"))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", install)
    monkeypatch.setattr("fast_browser_use.model.get_model", model)
    monkeypatch.setattr(sys, "argv", ["fbu", "install-browser"])
    if returncode:
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == returncode
    else:
        cli.main()
    install.assert_called_once_with(
        [sys.executable, "-m", "playwright", "install", "chromium"], check=False,
    )
    model.assert_not_called()


def test_bare_invocation_does_not_start_the_inspector(monkeypatch):
    serve = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.demo.serve", serve)
    monkeypatch.setattr(sys, "argv", ["fbu"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    serve.assert_not_called()
    assert exit_info.value.code not in (0, None)


def test_explicit_serve_starts_the_inspector(monkeypatch):
    serve = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.demo.serve", serve)
    monkeypatch.setattr(sys, "argv", ["fbu", "serve", "--port", "9999"])
    cli.main()
    serve.assert_called_once_with(9999)


def test_modelscope_converted_repository_is_downloaded_directly(monkeypatch, tmp_path):
    download = Mock()
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fbu", "download", "--backend", "mlx", "--source", "modelscope",
            "--revision", "pinned-revision", "--output", str(tmp_path),
        ],
    )
    cli.main()
    download.assert_called_once_with(
        "mlx-community/Qwen3.5-9B-4bit", revision="pinned-revision", local_dir=str(tmp_path),
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )


@pytest.mark.parametrize("arguments", [
    ["--url", "https://example.test", "--goal", "Find a result"],
    ["--goal", "Find a result", "--expect-title", "Result"],
    ["--scenario", "research", "--url", "https://example.test", "--goal", "Find", "--expect-title", "Result"],
    ["--scenario", "research", "--expect-title", "Result"],
])
def test_custom_recording_requires_goal_url_and_assertions_before_loading(monkeypatch, arguments):
    record = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.recording.record", record)
    monkeypatch.setattr(sys, "argv", ["fbu", "record", *arguments])
    with pytest.raises(SystemExit):
        cli.main()
    record.assert_not_called()


def test_custom_recording_forwards_outcome_separately(monkeypatch):
    record = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.recording.record", record)
    monkeypatch.setattr(sys, "argv", [
        "fbu", "record", "--url", "https://example.test", "--goal", "Find a result",
        "--expect-title", "Result", "--expect-text", "First", "--expect-text", "Second",
    ])
    cli.main()
    record.assert_called_once_with(
        None, scenario=None, url="https://example.test", goal="Find a result",
        expected={"url": None, "title": "Result", "text": ["First", "Second"]},
    )


@pytest.mark.parametrize("title, passed", [("Result", True), ("Still loading", False)])
def test_run_independently_rejects_false_done_and_retains_trace(monkeypatch, tmp_path, title, passed):
    import json

    class FakeAgent:
        def __init__(self, url, goal):
            assert (url, goal) == ("https://example.test", "Find a result")
            self.state = {"status": "done"}
            self.browser = SimpleNamespace(evaluate=Mock(return_value={
                "url": "https://example.test", "title": title, "text": "",
            }))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self):
            return iter(())

        def snapshot(self):
            return dict(self.state)

    trace = tmp_path / "result.json"
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.agent.Agent", FakeAgent)
    monkeypatch.setattr("fast_browser_use.model.get_model", Mock())
    monkeypatch.setattr(sys, "argv", [
        "fbu", "run", "https://example.test", "--goal", "Find a result",
        "--expect-title", "Result", "--trace", str(trace),
    ])
    if passed:
        cli.main()
    else:
        with pytest.raises(SystemExit, match="assertions failed"):
            cli.main()
    saved = json.loads(trace.read_text())
    assert saved["status"] == "done" and saved["verification"]["passed"] == passed


@pytest.mark.parametrize("source", ["mlx", "modelscope"])
def test_download_defaults_to_pinned_qwen_without_conversion(monkeypatch, source):
    from fast_browser_use.model import DEFAULT_MODEL, DEFAULT_REVISION, MODELSCOPE_REVISION

    download = Mock(return_value="/cached/qwen")
    module = "huggingface_hub" if source == "mlx" else "modelscope"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fbu", "download", "--backend", "mlx", "--source", source])
    cli.main()
    download.assert_called_once_with(
        DEFAULT_MODEL, revision=DEFAULT_REVISION if source == "mlx" else MODELSCOPE_REVISION,
        local_dir=None if source == "mlx" else "models/Qwen3.5-9B-4bit",
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )


def test_install_browser_can_install_linux_dependencies(monkeypatch):
    install = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", install)
    monkeypatch.setattr(sys, "argv", ["fbu", "install-browser", "--with-deps"])
    cli.main()
    install.assert_called_once_with(
        [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=False,
    )


def test_torch_download_uses_original_pinned_weights_without_inference(monkeypatch):
    from fast_browser_use.model import TORCH_MODEL, TORCH_REVISION

    download = Mock(return_value="/cached/qwen")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fbu", "download", "--backend", "torch"])
    cli.main()
    download.assert_called_once_with(
        TORCH_MODEL, revision=TORCH_REVISION, local_dir=None,
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )


@pytest.mark.parametrize("source", ["mlx", "modelscope"])
def test_incompatible_torch_download_source_fails_before_downloading(monkeypatch, source):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fbu", "download", "--backend", "torch", "--source", source])
    with pytest.raises(SystemExit):
        cli.main()
