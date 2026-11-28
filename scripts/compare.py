"""Print scorecards side by side as a Markdown table, for the README.

    python -m scripts.compare runs/baseline-agent.json runs/baseline-pipeline.json
"""
import json
import sys

cards = [json.load(open(p)) for p in sys.argv[1:]]
keys = list(cards[0]["scorecard"])
print("| metric | " + " | ".join(c["label"] for c in cards) + " |")
print("|---|" + "---|" * len(cards))
for k in keys:
    print(f"| {k} | " + " | ".join(str(c["scorecard"][k]) for c in cards) + " |")