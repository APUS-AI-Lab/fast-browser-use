"""Loopback inspector. A single worker owns Playwright and serializes browser mutations."""

import json
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import Agent
from .model import DEFAULT_MODEL, get_model
from .questions import MAX_STEPS

ROOT = Path(__file__).parent
GOALS = {
    "research": "Open the article about using finite choices to control browser agents.",
    "flights": "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. "
    "Stop when matching flight options are visible. Do not select or book a flight.",
}


def load_environment():
    path = Path.cwd() / ".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Inspector:
    def __init__(self, origin):
        self.origin, self.agent = origin, None

    def state(self):
        state = (
            self.agent.snapshot()
            if self.agent
            else {
                "page": None,
                "status": "idle",
                "history": [],
                "decision": None,
            }
        )
        return {**state, "text_model": os.environ.get("FBU_MODEL", DEFAULT_MODEL), "max_steps": MAX_STEPS}

    def command(self, name, body):
        if name == "reset":
            scenario = body.get("scenario", "research")
            if scenario not in {*GOALS, "custom"}:
                raise ValueError("Unknown scenario")
            goal = body.get("goal", "").strip()
            if not 0 < len(goal) <= 4000:
                raise ValueError("Enter a goal of 1–4,000 characters")
            url = (
                body.get("url", "")
                if scenario == "custom"
                else (
                    "https://www.google.com/travel/flights?hl=en"
                    if scenario == "flights"
                    else f"{self.origin}/fixture.html?scenario={scenario}"
                )
            )
            if urlparse(url).scheme not in {"http", "https"}:
                raise ValueError("Enter an http:// or https:// URL")
            self.close()
            self.agent = Agent(url, goal, screenshots=True)
        elif self.agent is not None:
            self.agent.command(name, body)
        else:
            raise ValueError("Start a demo first")
        return self.state()

    def close(self):
        if self.agent:
            self.agent.close()
            self.agent = None


def serve(port=8767):
    token = secrets.token_urlsafe(32)
    origin = f"http://127.0.0.1:{port}"
    inspector = Inspector(origin)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser")
    # Exclude weight loading from the warm demo; disclose this measurement boundary.
    print("Loading local Qwen3.5-9B…", flush=True)
    worker.submit(get_model).result()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, mime="application/json"):
            content = content if isinstance(content, bytes) else content.encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{port}":
                return self.send(403, "Forbidden", "text/plain")
            path = urlparse(self.path).path
            if path == "/api/state":
                return self.send(200, json.dumps(worker.submit(inspector.state).result()))
            if path == "/demo.mp4":
                video = ROOT.parent / "docs/demo.mp4"
                if video.exists():
                    return self.send(200, video.read_bytes(), "video/mp4")
            files = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
                "/fixture.html": ("fixture.html", "text/html"),
            }
            if path not in files:
                return self.send(404, "Not found", "text/plain")
            name, mime = files[path]
            self.send(200, (ROOT / "static" / name).read_text().replace("__TOKEN__", token), mime + "; charset=utf-8")

        def do_POST(self):
            if (
                self.headers.get("Host") != f"127.0.0.1:{port}"
                or self.headers.get("X-Demo-Token") != token
                or self.headers.get("Origin") not in (None, origin)
            ):
                return self.send(403, json.dumps({"error": "Local demo requests only"}))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 16384:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                result = worker.submit(inspector.command, self.path.removeprefix("/api/"), body).result()
                self.send(200, json.dumps(result))
            except Exception as error:
                self.send(400, json.dumps({"error": str(error)}))

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Loopback inspector (debug only): {origin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.submit(inspector.close).result()
        worker.shutdown()
