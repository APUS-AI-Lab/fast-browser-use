"""Package a verified run; label sped-up previews and retain the original recording."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("--name", default="demo", help="Output asset basename, e.g. wikipedia")
parser.add_argument("--max-seconds", type=float, help="Fit preview using labeled integer playback speed; no cuts")
args = parser.parse_args()
if not args.name.replace("-", "").replace("_", "").isalnum():
    parser.error("--name must contain only letters, numbers, underscores or hyphens")
if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds < 1):
    parser.error("--max-seconds must be finite and at least 1")
source = args.source.resolve()
summary = json.loads((source / "summary.json").read_text())
trace = json.loads((source / "trace.json").read_text())
if not summary["verified"] or not trace["verification"]["passed"] or summary["error"] or trace["status"] != "done":
    raise SystemExit("Refusing to publish an unverified run as a successful demonstration")
root = Path(__file__).resolve().parents[1]
docs = root / "docs"
docs.mkdir(exist_ok=True)


def duration(path):
    result = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], text=True
    )
    return float(json.loads(result)["format"]["duration"])


raw_seconds = duration(source / "browser.webm")
speed = math.ceil(raw_seconds / (args.max_seconds - 0.1)) if args.max_seconds else 1


def font(size, bold=False):
    candidates = [
        f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# Preserve the complete time interval. An accelerated preview gets a separate original file.
stream = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
    "-of", "json", str(source / "browser.webm"),
], text=True))["streams"][0]
width, height = stream["width"], stream["height"]
side = width + 65
canvas = Image.new("RGB", (width + 392, height + 256), "#f4f5ed")
draw = ImageDraw.Draw(canvas)
ink, green, muted = "#172820", "#286144", "#707c70"
draw.text((32, 28), "fast / browser", font=font(22, True), fill=ink)
local_page = urlparse(trace["page"]["url"]).hostname in {"localhost", "127.0.0.1", "::1"}
kind = "LOCAL FIXTURE" if local_page else "LIVE WEBSITE"
draw.text((width, 32), f"{kind}  /  {speed}x PLAYBACK", font=font(17, True), fill=green)
if trace["scenario"] == "wikipedia":
    draw.text((30, 79), "Wikipedia: find and open an article", font=font(40, True), fill=ink)
    subtitle = "Python, the programming language | Real website | Independently verified"
else:
    draw.text((30, 79), "One goal. Entirely local.", font=font(49, True), fill=ink)
    subtitle = "The local model chooses the actions and generates the words. No inference API."
draw.text((33, 145), subtitle, font=font(21), fill=muted)
draw.rounded_rectangle((30, 192, width + 34, height + 196), radius=10, fill="#d4ddce")
draw.text((side, 222), "THIS VERIFIED RUN", font=font(15, True), fill=green)
draw.text((side - 5, 262), f"{summary['task_ms'] / 1000:.2f}s", font=font(53, True), fill=ink)
draw.text((side, 328), "agent execution time", font=font(18), fill=muted)
decision_ms = summary['median_action_decision_ms']
draw.text((side - 2, 409), f"{decision_ms:.0f} ms" if decision_ms else "n/a", font=font(36, True), fill=ink)
draw.text((side, 458), "median action decision", font=font(18), fill=muted)
draw.text((side, 533), f"{summary['actions']} browser actions", font=font(23, True), fill=ink)
draw.text((side, 574), f"{summary['scoring_calls']} one-token passes", font=font(20), fill=muted)
draw.text((side, 612), f"{summary['text_calls']} local text call(s)", font=font(20), fill=muted)
draw.text((side, 650), f"{summary.get('plan_calls', 0)} local planning calls", font=font(20), fill=muted)
if summary.get("reasoning_calls"):
    draw.text((side, 679), f"{summary['reasoning_calls']} deliberation calls", font=font(18), fill=muted)
draw.text((side, 705), trace.get("model_label", "QWEN3.5 9B"), font=font(20, True), fill=green)
inference = trace.get("inference", {"backend": "mlx", "dtype": "4-bit", "device": "metal"})
draw.text((side, 744), f"{inference['backend'].upper()} / {inference['dtype'].removeprefix('torch.')}",
          font=font(20), fill=muted)
draw.text((side, 797), trace["hardware"].get("chip", trace["hardware"].get("machine", "Unknown")),
          font=font(20), fill=ink)
memory_gb = trace["hardware"].get("memory_bytes", 0) // 1024**3
draw.text((side, 835), f"{memory_gb} GB memory" if memory_gb else f"Device: {inference['device']}",
          font=font(18), fill=muted)
draw.text(
    (33, height + 218),
    ("Original timing. " if speed == 1 else f"{speed}x playback; original recording {raw_seconds:.2f}s. ")
    + "All inference and waits included. Model loaded before the task.",
    font=font(17),
    fill=muted,
)
frame_path = source / "frame.png"
canvas.save(frame_path)
filters = "[1:v][0:v]overlay=32:194:shortest=1"
extra_inputs = []
if speed > 1:
    # The original recorder's HUD says ORIGINAL SPEED. Replace that label in the preview.
    badge = Image.new("RGBA", canvas.size)
    badge_draw = ImageDraw.Draw(badge)
    top = 194 + height - 40
    badge_draw.rectangle((32, top, 32 + width, top + 40), fill="#172820")
    badge_draw.text(
        (56, top + 11),
        f"LOCAL MODEL  |  {speed}x PLAYBACK  |  REAL TASK {summary['task_ms'] / 1000:.2f}s  |  OUTCOME VERIFIED",
        font=font(16, True), fill="#c9f5a0",
    )
    badge_path = source / "playback-label.png"
    badge.save(badge_path)
    extra_inputs = ["-loop", "1", "-i", str(badge_path)]
    filters += "[framed];[framed][2:v]overlay=0:0:shortest=1"
    shutil.copyfile(source / "browser.webm", docs / f"{args.name}-original.webm")
filters += f",setpts=(PTS-STARTPTS)/{speed}[out]"
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source / "browser.webm"),
        "-loop",
        "1",
        "-i",
        str(frame_path),
        *extra_inputs,
        "-filter_complex",
        filters,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(docs / f"{args.name}.mp4"),
    ],
    check=True,
)
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(docs / f"{args.name}.mp4"),
        "-vf",
        "fps=10,scale=1050:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
        "-loop",
        "0",
        str(docs / f"{args.name}.gif"),
    ],
    check=True,
)
shutil.copyfile(source / "result.png", docs / f"{args.name}-result.png")
evidence = {
    k: trace[k]
    for k in (
        "goal",
        "elapsed_ms",
        "verification",
        "source_hashes",
        "model",
        "model_revision",
        "runtime",
        "hardware",
        "model_load_ms",
        "text_calls",
        "timing_boundary",
        "error",
        "scenario",
        "plan",
        "plan_calls",
        "milestones",
    )
}
evidence["summary"] = {k: v for k, v in summary.items() if k != "folder"}
evidence["request"] = trace.get("request")
evidence["final_page"] = {k: trace["page"][k] for k in ("url", "title")}
evidence["recorded_at"] = trace.get("recorded_at")
evidence["local_deliberation"] = trace.get("local_deliberation", False)
evidence["configuration"] = trace.get("configuration")
evidence["inference"] = inference
evidence["rejections"] = trace.get("rejections", [])
evidence["observations"] = trace.get("observations", [])
evidence["browser_locale"] = trace.get("browser_locale")
evidence["browser_headless"] = trace.get("browser_headless")
evidence["decisions"] = [
    {k: d.get(k) for k in (
        "operation", "target", "latency_ms", "usage", "cache_hit", "completion_check", "reasoning", "scoring_passes",
        "prefill_step_size",
    )}
    for d in trace["decisions"]
]
for decision in evidence["decisions"]:
    if decision.get("reasoning"):
        decision["reasoning"] = {k: v for k, v in decision["reasoning"].items() if k != "text"}
    if (decision.get("completion_check") or {}).get("reasoning"):
        decision["completion_check"] = {
            **decision["completion_check"], "reasoning": decision["reasoning"],
        }
evidence["actions"] = [
    {k: h[k] for k in ("action", "kind", "text", "latency_ms", "executed_ms", "text_latency_ms")}
    for h in trace["history"]
]
evidence["video"] = {
    "speed": speed,
    "cuts": 0,
    "final_hold_seconds": 1,
    "includes": "Initial page setup, all task time, verification, and a final hold; no cuts",
    "original_file": f"{args.name}-original.webm" if speed > 1 else None,
}


mp4_seconds = duration(docs / f"{args.name}.mp4")
if abs(raw_seconds / speed - mp4_seconds) > 0.1:
    raise SystemExit("Rendered video duration differs from the declared playback speed")
if args.max_seconds and mp4_seconds > args.max_seconds:
    raise SystemExit("Rendered preview exceeds --max-seconds")
evidence["video"].update(
    raw_seconds=raw_seconds,
    mp4_seconds=mp4_seconds,
    raw_sha256=hashlib.sha256((source / "browser.webm").read_bytes()).hexdigest(),
    mp4_sha256=hashlib.sha256((docs / f"{args.name}.mp4").read_bytes()).hexdigest(),
)
measurement_name = "measurement.json" if args.name == "demo" else f"{args.name}-measurement.json"
(docs / measurement_name).write_text(json.dumps(evidence, indent=2))
print(f"Published {speed}x playback assets in", docs)
