"""Isolated Chromium with guarded DOM actions, adapted from Jev Ultrafast."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"


class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url, *, video_dir=None, viewport=None):
        self.driver = sync_playwright().start()
        self.chrome = None
        try:
            self.headless = os.environ.get("FBU_HEADLESS", "1") != "0"
            self.chrome = self.driver.chromium.launch(headless=self.headless)
            self.locale = os.environ.get("FBU_LOCALE", "en-US")
            options = {
                "viewport": viewport or {"width": 1120, "height": 780},
                "device_scale_factor": 1, "locale": self.locale,
            }
            if video_dir:
                options.update(record_video_dir=str(video_dir), record_video_size=options["viewport"])
            self.context = self.chrome.new_context(**options)
            self.page = self.context.new_page()
            self.navigations = []
            self.page.on("request", self._navigation_request)
            self.video = self.page.video
            self.session = self.context.new_cdp_session(self.page)
            self.target = "isolated-chromium"
            self.call("Emulation.setFocusEmulationEnabled", enabled=True)
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception:
            self.close()
            raise

    def call(self, method, **params):
        return self.session.send(method, params)

    def _navigation_request(self, request):
        if request.is_navigation_request() and request.frame == self.page.main_frame:
            self.navigations.append({"url": request.url, "method": request.method, "time": time.time()})

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot=True):
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            # This is read-only and happens after execution was logged, even if navigation interrupts it.
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__fastBrowserUse?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))("""
                    + json.dumps(action)
                    + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except RuntimeError:
                pass
        # Live sites can redirect for longer than a few animation frames. Retry only
        # this read; input has already been recorded and must never be replayed.
        for attempt in range(600):
            try:
                state = browser_operation({"operation": "observe", "session": self.session, "screenshot": screenshot})
                if not state["text"].strip() and not any(a.get("node") for a in state["actions"]) and attempt < 599:
                    raise StalePage("Waiting for the initial document to render")
                return state
            except StalePage:
                if attempt == 599:
                    raise
                time.sleep(0.05)
        raise StalePage("Page did not settle")

    def fresh(self, page, action=None):
        if action is not None and action["kind"] == "scroll" and action.get("node") is not None:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__fastBrowserUse; "
                f"return c ? [c.pageKey(),c.scrollGuard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], action["scroll_state"]]
        if action is not None and action["kind"] in {"click", "select", "fill"}:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__fastBrowserUse; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def prepare(self, page, *, screenshot=False):
        """Bounded, read-only settling before inference; never weaken dispatch guards.

        Compare the observed semantic state, not arbitrary DOM mutations (e.g. a
        recording clock). Wait a little longer after filling so debounced search
        suggestions can appear, including searchboxes without role=combobox.
        """
        quiet_ms = int(os.environ.get("FBU_SETTLE_MS", "150"))
        if not 0 <= quiet_ms <= 2000:
            raise ValueError("FBU_SETTLE_MS must be between 0 and 2000")
        if quiet_ms == 0:
            return page
        started = changed = time.perf_counter()
        minimum = max(quiet_ms / 1000, 0.3 if getattr(self, "last_input_kind", None) == "fill" else 0)
        document = page["page_key"][0]
        if document != getattr(self, "prepared_document", None):
            # DOMContentLoaded can precede hydration of menus and controls. This
            # bounded window is inside the task timer, including the first page.
            minimum = max(minimum, 0.5)
        self.prepared_document = document
        self.last_input_kind = None
        deadline = started + max(1.5, minimum * 4)
        previous, samples, settled = page.get("marker"), 0, False
        while True:
            try:
                observed = browser_operation({"operation": "observe", "session": self.session, "screenshot": False})
            except StalePage:
                # Only the read is repeated. Any input was already logged.
                previous, changed = None, time.perf_counter()
            else:
                page = observed
                samples += 1
                now = time.perf_counter()
                has_content = bool(page["text"].strip()) or any(a.get("node") for a in page["actions"])
                ready = page.get("ready", True) and has_content
                if page["marker"] != previous or not ready:
                    previous, changed = page["marker"], now
                if ready and now - changed >= quiet_ms / 1000 and now - started >= minimum:
                    settled = True
                    break
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.05)
        if not samples:
            raise StalePage("Document changed throughout preparation; observe again before inference")
        if screenshot:
            page["screenshot"] = self.call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
        page["preparation"] = {
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "samples": samples,
            "settled": settled,
        }
        return page

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        result = browser_operation({"operation": "act", "session": self.session, "action": action, "text": text})
        self.after_input = action if action["kind"] != "wait" else None
        self.last_input_kind = action["kind"]
        return result

    def close(self, *, video_path=None):
        try:
            if self.chrome is not None:
                try:
                    if getattr(self, "context", None):
                        self.context.close()
                    if video_path and getattr(self, "video", None):
                        self.video.save_as(str(video_path))
                finally:
                    self.chrome.close()
                    self.chrome = None
        finally:
            self.driver.stop()


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return session.send(method, params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            node = action.get("node")
            if node is not None and type(node) is not int:
                raise ValueError("Invalid observed scroll region")
            target = evaluate(
                "(() => { const c=window.__fastBrowserUse; return c?.scrollPoint("
                + (f"c.nodes.get({node})" if node is not None else "document.scrollingElement") + "); })()"
            )
            if target is None:
                raise StalePage("Scroll region is covered or no longer available")
            call("Input.dispatchMouseEvent", type="mouseWheel", **target, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate(
                """(action => {
              const e=window.__fastBrowserUse?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (e.tagName==='LABEL' && (!e.control || e.control.matches(':disabled') ||
                  e.control.closest('[aria-disabled="true"]'))) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })("""
                + json.dumps(action)
                + ")"
            )
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    call("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
