"""Run URL/goal/outcome task manifests with local inference; retain failures as well as successes.

Only --serve opens a loopback fixture server. No task-specific action logic or field strings.
Each repeat is a fresh browser run, so use repeated trials only on disposable/read-only tasks.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import statistics
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fast_browser_use import Agent
from fast_browser_use.demo import ROOT
from fast_browser_use.model import get_model
from fast_browser_use.recording import runtime_versions
from fast_browser_use.verification import validate_expectations, verify_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path, help="JSON list of {name, url, goal, expected: {url?, title?, text?}}")
    parser.add_argument("--serve", type=Path, help="Serve fixtures here; replace {base_url} in the manifest")
    parser.add_argument("--runs", type=int, default=1, help="Fresh trials per task; never an automatic mutation retry")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")
    folder = args.output or Path("artifacts/tasks") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=False)
    with contextlib.ExitStack() as stack:
        source = args.tasks.read_text()
        if args.serve:
            handler = partial(SimpleHTTPRequestHandler, directory=str(args.serve.resolve()))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            source = source.replace("{base_url}", f"http://127.0.0.1:{server.server_port}")
        tasks = json.loads(source)
        if not tasks or not isinstance(tasks, list):
            parser.error("Task manifest must be a nonempty list")
        for task in tasks:
            if not all(task.get(key) for key in ("name", "url", "goal", "expected")):
                parser.error("Every task needs name, url, goal and expected outcome assertions")
            if set(task["expected"]) - {"url", "title", "text"} or not any(task["expected"].values()):
                parser.error("Expected outcome supports nonempty url, title and/or text assertions")
            validate_expectations(**task["expected"])
        engine = get_model()
        rows = []
        for task_index, task in enumerate(tasks):
            for trial in range(1, args.runs + 1):
                result, error = {}, None
                try:
                    with Agent(task["url"], task["goal"]) as agent:
                        try:
                            for _state in agent.run():
                                pass
                        finally:
                            result = agent.snapshot()
                            result["verification"] = verify_outcome(agent.browser, **task["expected"])
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                result["error"] = error
                (folder / f"task-{task_index + 1}-trial-{trial}.json").write_text(json.dumps(result, indent=2))
                row = {
                    "task": task["name"], "trial": trial, "task_ms": result.get("elapsed_ms"),
                    "status": result.get("status"), "actions": len(result.get("history", [])),
                    "passed": not error and result.get("status") == "done"
                    and result.get("verification", {}).get("passed", False),
                    "verification": result.get("verification"), "error": error,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = {
            "model": engine.name, "model_revision": engine.revision,
            "runtime": runtime_versions(),
            "platform": platform.platform(),
            "source_hashes": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in ROOT.rglob("*") if path.suffix in {".py", ".js", ".html", ".css"}
            },
            "configuration": {
                "planning": os.environ.get("FBU_PLAN", "0") != "0",
                "decision_mode": os.environ.get("FBU_DECISION_MODE", "auto"),
                "settle_ms": int(os.environ.get(
                    "FBU_SETTLE_MS", "150" if os.environ.get("FBU_PLAN", "0") == "0" else "0",
                )),
            },
            "runs": rows, "passed": sum(row["passed"] for row in rows), "total": len(rows),
            "median_task_ms": statistics.median(row["task_ms"] for row in rows if row["task_ms"] is not None)
            if any(row["task_ms"] is not None for row in rows) else None,
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        print(folder, flush=True)
        if summary["passed"] != summary["total"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
