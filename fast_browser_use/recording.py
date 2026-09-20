"""Record the browser's actual frames at original speed and verify the final DOM independently."""

import contextlib
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .agent import Agent
from .demo import GOALS, ROOT
from .model import get_model
from .verification import validate_expectations, verify_flights, verify_outcome

HUD = """(() => {
 const attach=()=>{
 if(document.getElementById('fbu-recording')) return;
 const panel=document.createElement('div');panel.id='fbu-recording';panel.setAttribute('aria-hidden','true');
 panel.style.cssText='position:fixed;bottom:0;left:0;right:0;height:40px;background:#172820;color:#c9f5a0;'
   +'font:14px monospace;display:flex;align-items:center;justify-content:space-between;padding:0 24px;'
   +'pointer-events:none;z-index:2147483647;box-sizing:border-box';
 document.documentElement.append(panel);
 const start=__EPOCH__;window.__fbuRecording={actions:0,ms:null,verified:false};
 setInterval(()=>{const s=window.__fbuRecording;
 panel.textContent=__MODEL_LABEL__+'  ·  ORIGINAL SPEED  ·  '
 +((s.ms??(performance.timeOrigin+performance.now()-start))/1000).toFixed(2)
 +'s  ·  '+s.actions+' actions'+(s.verified?'  ·  OUTCOME VERIFIED':'');},30);
 };
 if(document.documentElement) attach();else addEventListener('DOMContentLoaded',attach,{once:true});
})()"""


def runtime_versions():
    installed = {}
    for name in ("mlx", "mlx-lm", "torch", "accelerate", "transformers", "playwright"):
        with contextlib.suppress(PackageNotFoundError):
            installed[name] = version(name)
    return installed


def record(output=None, *, scenario=None, url=None, goal=None, expected=None):
    custom = url is not None or goal is not None
    if custom and (not url or not goal or scenario or not expected or not any(expected.values())):
        raise ValueError("A custom recording needs URL, goal and outcome assertions, without a scenario")
    if not custom and expected:
        raise ValueError("Outcome assertions require a custom URL and goal")
    if custom:
        validate_expectations(**expected)
    scenario = "custom" if custom else scenario or "research"
    folder = Path(output or "artifacts/recordings/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True, exist_ok=False)
    engine = get_model()
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob("*") if p.suffix in {".py", ".js", ".html", ".css"}
    }

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = None
    if scenario == "research":
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "static")))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/fixture.html?scenario={scenario}"
    if not custom:
        goal = GOALS.get(scenario)
    if scenario == "wikipedia":
        url = "https://en.wikipedia.org/wiki/Main_Page"
        goal = "Find and open the Wikipedia article about Python, the programming language."
    elif scenario == "flights":
        url = "https://www.google.com/travel/flights?hl=en"
    try:
        # Playwright captures Chromium frames without a desktop, DISPLAY or Xvfb.
        agent = Agent(url, goal, video_dir=folder / "raw", headless=True)
    except BaseException:
        if server:
            server.shutdown()
            server.server_close()
        raise
    error = None
    model_label = "LOCAL " + engine.name.rsplit("/", 1)[-1].replace("-it-4bit", "").replace("-", " ").upper()
    started = time.perf_counter()
    try:
        epoch = agent.browser.evaluate("performance.timeOrigin+performance.now()")
        hud = HUD.replace("__EPOCH__", str(epoch)).replace("__MODEL_LABEL__", json.dumps(model_label))
        agent.browser.context.add_init_script(hud)
        agent.browser.evaluate(hud)
        agent.state["started_at"] = time.perf_counter()
        started = agent.state["started_at"]
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            decision = state["decisions"][-1] if state["decisions"] else {}
            print(
                state["elapsed_ms"], state["status"], decision.get("operation", "OBSERVE"),
                f"checkpoint={state['plan_index']}", last.get("action", ""), flush=True,
            )
            (folder / "progress.json").write_text(json.dumps(state, indent=2))
            # Navigation may replace the document between observation and this cosmetic update.
            with contextlib.suppress(Exception):
                agent.browser.evaluate(
                    f"if(window.__fbuRecording) window.__fbuRecording.actions={len(state['history'])}"
                )
            if time.perf_counter() - started > 600:
                raise RuntimeError("Recording exceeded its 600-second diagnostic budget")
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        result = agent.snapshot()
        result["error"] = error
        if error:
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        try:
            if custom:
                verification = verify_outcome(agent.browser, **expected)
            elif scenario == "flights":
                verification = verify_flights(agent.browser.observe(screenshot=False))
            else:
                facts = agent.browser.evaluate(
                    "({url:location.href,title:document.title,text:document.body.innerText})"
                )
                if scenario == "wikipedia":
                    checks = {
                        "article_url": facts["url"].split("?")[0]
                        == "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "article_title": facts["title"] == "Python (programming language) - Wikipedia",
                    }
                else:
                    checks = {
                        "article_url": facts["url"].endswith("#choices"),
                        "article_title": facts["title"] == "A browser is a choice, not a conversation · Forma",
                    }
                verification = {"passed": all(checks.values()), "checks": checks}
        except Exception as exc:
            verification = {"passed": False, "checks": {}, "error": f"{type(exc).__name__}: {exc}"}
        result["verification"] = verification
        result["scenario"] = scenario
        result["recorded_at"] = datetime.now(timezone.utc).isoformat()
        result["hardware"] = {"machine": platform.machine(), "platform": platform.platform()}
        if platform.system() == "Darwin":
            result["hardware"].update(
                chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()),
            )
        result["runtime"] = runtime_versions()
        result["inference"] = {"backend": engine.backend, "device": str(engine.device), "dtype": str(engine.dtype)}
        result["model"] = engine.name
        result["model_label"] = model_label.removeprefix("LOCAL ")
        result["model_revision"] = engine.revision
        result["browser_locale"] = agent.browser.locale
        result["browser_headless"] = agent.browser.headless
        result["local_deliberation"] = any(d.get("reasoning") for d in result["decisions"])
        result["model_load_ms"] = engine.load_ms
        result["configuration"] = {
            "planning": os.environ.get("FBU_PLAN", "0") != "0",
            "decision_mode": os.environ.get("FBU_DECISION_MODE", "auto"),
            "settle_ms": int(os.environ.get(
                "FBU_SETTLE_MS", "150" if os.environ.get("FBU_PLAN", "0") == "0" else "0",
            )),
        }
        result["timing_boundary"] = (
            "Warm model, after initial navigation/observation; all planning, decisions, text, actions, waits."
        )
        result["source_hashes"] = source_hashes
        result["navigations"] = agent.browser.navigations
        verified = result["verification"]["passed"]
        result["recording_warnings"] = []
        try:
            agent.browser.evaluate(
                "if(window.__fbuRecording) Object.assign(window.__fbuRecording,"
                + json.dumps({"ms": result["elapsed_ms"], "actions": len(result["history"]), "verified": verified})
                + ")"
            )
            agent.browser.page.wait_for_timeout(1000)  # Explicit final hold, outside the measured task.
            (folder / "result.png").write_bytes(agent.browser.page.screenshot())
            (folder / "result.html").write_text(agent.browser.page.content())
        except Exception as exc:
            result["recording_warnings"].append(str(exc))
        finally:
            (folder / "trace.json").write_text(json.dumps(result, indent=2))
            try:
                agent.browser.close(video_path=folder / "browser.webm")
            finally:
                if server:
                    server.shutdown()
                    server.server_close()
    latencies = [d["latency_ms"] for d in result["decisions"]]
    summary = {
        "task_ms": result["elapsed_ms"],
        "median_decision_ms": statistics.median(latencies) if latencies else None,
        "median_action_decision_ms": statistics.median(
            [d["latency_ms"] for d in result["decisions"] if d["operation"] not in {"DONE", "BLOCKED"}]
        )
        if result["history"]
        else None,
        "scoring_calls": sum(
            d.get("scoring_passes", 1 if d["operation"] == "DONE" else 2) for d in result["decisions"]
        ),
        "decisions": len(latencies),
        "rejected_decisions": sum(
            r["decision_index"] is not None and not r["execution_recorded"] for r in result.get("rejections", [])
        ),
        "preparation_ms": round(sum(o["latency_ms"] for o in result.get("observations", [])), 2),
        "actions": len(result["history"]),
        "text_calls": len(result["text_calls"]),
        "plan_calls": len(result["plan_calls"]),
        "planning_ms": sum(p["latency_ms"] for p in result["plan_calls"]),
        "reasoning_calls": sum(bool(d.get("reasoning")) for d in result["decisions"]),
        "reasoning_ms": sum(d["reasoning"]["latency_ms"] for d in result["decisions"] if d.get("reasoning")),
        "verified": verified,
        "error": error,
        "folder": str(folder),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if error or not verified or result["status"] != "done":
        raise RuntimeError(f"Run failed verification; retained evidence at {folder}")
    return folder
