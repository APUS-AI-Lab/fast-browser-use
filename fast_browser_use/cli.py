"""Local-only CLI, including an optional download/convert path from ModelScope."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .demo import load_environment


def main():
    load_environment()
    parser = argparse.ArgumentParser(description="Local Qwen3.5-9B browser-use skill. Host agents invoke `fbu run`.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a natural-language goal on a URL (skill entrypoint)")
    run.add_argument("url")
    run.add_argument("--goal", required=True)
    run.add_argument("--trace", default="artifacts/run.json")
    browser = sub.add_parser("install-browser", help="Install Chromium for this fbu environment (no model download)")
    browser.add_argument("--with-deps", action="store_true", help="Also install Linux browser system dependencies")
    download = sub.add_parser("download", help="Download weights for the selected local backend (no inference API)")
    download.add_argument("--source", choices=["huggingface", "mlx", "modelscope"], default="huggingface",
                          help="Model host; mlx is a legacy alias for Hugging Face MLX weights")
    download.add_argument("--revision", help="Pin a model repository revision")
    download.add_argument("--output", help="Local download directory (otherwise source-specific cache)")
    recording = sub.add_parser("record", help="Record and independently verify a task (original timing)")
    recording.add_argument(
        "--scenario", choices=["research", "wikipedia", "flights"]
    )
    recording.add_argument("--url", help="Start URL for a custom task; use with --goal and outcome assertions")
    recording.add_argument("--goal", help="Natural-language goal for a custom task")
    recording.add_argument("--output", help="New directory for raw recording and trace")
    for command in (run, recording):
        command.add_argument("--expect-url", help="Exact final URL to verify independently")
        command.add_argument("--expect-title", help="Exact final title to verify independently")
        command.add_argument("--expect-text", action="append", default=[], help="Required visible text; repeatable")
    web = sub.add_parser("serve", help="Optional loopback inspector for debugging candidate scores")
    web.add_argument("--port", type=int, default=int(os.environ.get("FBU_PORT", "8767")))
    for command in (run, recording, download, web):
        command.add_argument("--backend", choices=["auto", "mlx", "torch"], help="Overrides FBU_BACKEND")
        command.add_argument("--model", help="Model name, alias (9b, 35b) or local directory (FBU_MODEL)")
    for command in (run, recording, web):
        command.add_argument("--device", help="PyTorch device: auto, cpu, cuda or cuda:N (FBU_DEVICE)")
        command.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], help="PyTorch dtype")
    args = parser.parse_args()
    for option in ("backend", "device", "dtype", "model"):
        value = getattr(args, option, None)
        if value is not None:
            os.environ[f"FBU_{option.upper()}"] = value
    if args.command in {"run", "record"}:
        expected = {"url": args.expect_url, "title": args.expect_title, "text": args.expect_text}
        has_expectations = args.expect_url is not None or args.expect_title is not None or bool(args.expect_text)
        if has_expectations:
            from .verification import validate_expectations

            try:
                validate_expectations(**expected)
            except ValueError as exc:
                parser.error(str(exc))
    if args.command == "record":
        custom = args.url is not None or args.goal is not None
        if custom and (not args.url or not args.goal or args.scenario or not has_expectations):
            parser.error("Custom recording needs --url, --goal and at least one --expect-*; omit --scenario")
        if has_expectations and not custom:
            parser.error("--expect-* is for custom recordings with --url and --goal")
    if args.command == "install-browser":
        command = [sys.executable, "-m", "playwright", "install"]
        if args.with_deps:
            command.append("--with-deps")
        result = subprocess.run([*command, "chromium"], check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    elif args.command == "download":
        from .model import MODELSCOPE_REVISION, model_source, resolve_backend

        backend = resolve_backend()
        if args.source == "mlx":
            if args.backend == "torch":
                parser.error("--source mlx downloads MLX weights; use --source huggingface with --backend torch")
            backend = "mlx"
        try:
            name, revision = model_source(backend, args.model)
        except ValueError as exc:
            parser.error(str(exc))
        if args.source == "modelscope":
            if backend != "mlx":
                parser.error("The ModelScope mirror is pinned for MLX only; use --source huggingface for PyTorch")
            try:
                from modelscope import snapshot_download
            except ImportError:
                parser.error("Install the ModelScope extra: uv sync --extra modelscope")
            output = args.output or "models/Qwen3.5-9B-4bit"
            snapshot_download(
                name, revision=args.revision or MODELSCOPE_REVISION, local_dir=output,
                allow_patterns=["*.json", "*.jinja", "*.safetensors"],
            )
            print(f"Set FBU_MODEL={Path(output).resolve()}")
        else:
            from huggingface_hub import snapshot_download

            location = snapshot_download(
                name, revision=args.revision or revision,
                local_dir=args.output, allow_patterns=["*.json", "*.jinja", "*.safetensors"],
            )
            print(f"Set FBU_MODEL={Path(location).resolve()}" if args.output else location)
    elif args.command == "record":
        from .recording import record

        record(
            args.output, scenario=args.scenario,
            url=args.url, goal=args.goal, expected=expected if has_expectations else None,
        )
    elif args.command == "run":
        from .agent import Agent
        from .model import get_model
        from .verification import verify_outcome

        get_model()
        folder = Path(args.trace)
        folder.parent.mkdir(parents=True, exist_ok=True)
        with Agent(args.url, args.goal) as agent:
            try:
                for state in agent.run():
                    last = state["history"][-1] if state["history"] else {}
                    print(state["elapsed_ms"], state["status"], last.get("action", ""), flush=True)
            finally:
                result = agent.snapshot()
                if has_expectations:
                    try:
                        result["verification"] = verify_outcome(agent.browser, **expected)
                    except Exception as exc:
                        result["verification"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                folder.write_text(json.dumps(result, indent=2))
            if agent.state["status"] != "done":
                raise SystemExit("Agent stopped without reporting completion; inspect the trace")
            if has_expectations and not result["verification"]["passed"]:
                raise SystemExit("Outcome assertions failed; inspect the trace. Do not blindly rerun mutations.")
        print("Outcome assertions passed. Trace:" if has_expectations else
              "Agent reports done. Independently verify the outcome. Trace:", folder)
    elif args.command == "serve":
        from .demo import serve

        serve(args.port)
