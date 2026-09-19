"""Live Google Flights search. Uses local Qwen3.5-9B; never selects or books a flight."""

import argparse
import json
from pathlib import Path

from fast_browser_use import Agent
from fast_browser_use.verification import verify_flights as verify

URL = "https://www.google.com/travel/flights?hl=en"
GOALS = (
    "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. "
    "Stop when matching flight options are visible. Do not select or book a flight."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/flights/latest")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    from fast_browser_use.model import get_model

    get_model()
    agent = Agent(URL, GOALS)
    try:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(state["elapsed_ms"], state["status"], last.get("action", ""), flush=True)
    finally:
        state = agent.snapshot()
        state["verification"] = verify(state["page"])
        (folder / "state.json").write_text(json.dumps(state, indent=2))
        agent.close()
    print(json.dumps(state["verification"], indent=2))
    if not state["verification"]["passed"]:
        raise SystemExit("Final page did not satisfy the route/date checks")


if __name__ == "__main__":
    main()
