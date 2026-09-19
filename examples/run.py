"""uv run python examples/run.py --url URL --goal 'A narrow goal'"""

import argparse

from fast_browser_use import Agent
from fast_browser_use.model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--goal", action="append", required=True, help="Repeat for an ordered list of goals.")
args = parser.parse_args()

get_model()
with Agent(args.url, args.goal) as agent:
    for state in agent.run():
        print(f"{state['elapsed_ms']:>5} ms  {len(state['history'])} actions  {state['status']}")
    print(state["page"]["url"])
