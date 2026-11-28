"""Ask the running API every golden question and write a scorecard.

    python -m scripts.eval --label baseline-agent --mode agent
    python -m scripts.eval --label baseline-pipeline --mode pipeline

Grading (fixed, deterministic, the golden set's own rules):
  correct      an `accept` string appears in the answer (lowercased, commas removed)
  evidence     a cited source is the expected doc and one of the expected pages
  false answer an unanswerable question that got an answer instead of a refusal
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

import httpx

GOLDEN = Path("starter/data/golden_set.json")
RUNS = Path("runs")


def ask(api: str, question: str, mode: str) -> dict:
    """The only part that knows our API. Swap it to evaluate another app."""
    r = httpx.post(f"{api}/chat", json={"question": question, "mode": mode},
                   headers={"X-User-Id": "eval"}, timeout=300)
    r.raise_for_status()
    return r.json()


def accepted(answer: str, accept: list[str]) -> bool:
    text = answer.lower().replace(",", "")
    return any(a.lower().replace(",", "") in text for a in accept)


def grade(q: dict, a: dict) -> dict:
    cited = {(s["doc"], s["page"]) for s in a["sources"]}
    return {
        "id": q["id"], "question": q["q"], "answerable": q["answerable"],
        "expected": q["answer"], "answer": a["answer"], "refused": a["refused"],
        "correct": accepted(a["answer"], q["accept"]) if q["answerable"] else None,
        "evidence": any(d == q["doc"] and p in q["pages"] for d, p in cited)
                    if q["answerable"] else None,
        "false_answer": (not a["refused"]) if not q["answerable"] else None,
        "citations": len(a["sources"]), "steps": len(a["steps"]),
        "seconds": a["seconds"], "usd": a["usage"]["usd"],
        "tokens": a["usage"]["prompt_tokens"] + a["usage"]["completion_tokens"],
    }


def scorecard(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["answerable"]]
    no = [r for r in rows if not r["answerable"]]
    pct = lambda xs: round(100 * mean(xs), 1) if xs else None
    return {
        "questions": len(rows),
        "accuracy %": pct([r["correct"] for r in ok]),
        "correct": f"{sum(r['correct'] for r in ok)}/{len(ok)}",
        "evidence cited %": pct([r["evidence"] for r in ok]),
        "refusals on answerable %": pct([r["refused"] for r in ok]),
        "correct refusals": f"{sum(not r['false_answer'] for r in no)}/{len(no)}",
        "false answers %": pct([r["false_answer"] for r in no]),
        "avg steps": round(mean(r["steps"] for r in rows), 2),
        "avg seconds": round(mean(r["seconds"] for r in rows), 2),
        "total usd": round(sum(r["usd"] for r in rows), 4),
        "avg tokens": round(mean(r["tokens"] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", default="agent", choices=["agent", "pipeline"])
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    questions = json.loads(GOLDEN.read_text())["questions"][: args.limit]
    rows = []
    for q in questions:
        try:
            a = ask(args.api, q["q"], args.mode)
        except Exception as e:  # one broken question must not lose the whole run
            a = {"answer": f"ERROR: {e}", "refused": True, "sources": [], "steps": [],
                 "seconds": 0, "usage": {"usd": 0, "prompt_tokens": 0, "completion_tokens": 0}}
        row = grade(q, a)
        rows.append(row)
        if q["answerable"]:
            mark = "✓" if row["correct"] else "✗"
        else:
            mark = "✓ refused" if row["refused"] else "✗ answered"
        print(f"{q['id']:>3} {mark:<10} steps={row['steps']} {row['seconds']:>5}s  {q['q'][:70]}")
        time.sleep(0.5)

    card = scorecard(rows)
    RUNS.mkdir(exist_ok=True)
    out = {"label": args.label, "mode": args.mode, "time": time.strftime("%Y-%m-%d %H:%M"),
           "scorecard": card, "rows": rows}
    (RUNS / f"{args.label}.json").write_text(json.dumps(out, indent=2))
    print("\n" + "\n".join(f"{k:<26} {v}" for k, v in card.items()))
    print(f"\nwritten: runs/{args.label}.json")


if __name__ == "__main__":
    main()