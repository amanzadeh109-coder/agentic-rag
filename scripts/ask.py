import sys

from app.agent import run_agent, run_pipeline

mode = sys.argv[2] if len(sys.argv) > 2 else "agent"
runner = run_agent if mode == "agent" else run_pipeline
for ev in runner(sys.argv[1]):
    if ev["type"] == "step":
        print(f"\n[step {ev['step']}] {ev['tool']}({ev['args']}) -> {ev['results']} results")
    elif ev["type"] == "token":
        print(ev["text"], end="", flush=True)
    else:
        print(f"\n\nsources: {[(s['file'], s['page'], s['verified']) for s in ev['sources']]}")
        print(f"refused: {ev['refused']} | {ev['usage']} | {ev['seconds']}s | {ev['model']}")