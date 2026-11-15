"""Print a run as a tree, plus totals per agent, guardrail hits and parallelism.

    python trace.py runs/<run_id>        # or the steps.jsonl file
    python trace.py latest               # the most recent run
"""
import sys
from collections import defaultdict
from pathlib import Path

from agent.evals import max_parallel
from agent.logger import format_event, read_log

sys.stdout.reconfigure(errors="replace")


def find_log(arg: str) -> Path:
    if arg == "latest":
        runs = [p for p in Path("runs").rglob("steps.jsonl")]
        if not runs:
            sys.exit("no runs yet")
        return max(runs, key=lambda p: p.stat().st_mtime)
    path = Path(arg)
    return path if path.is_file() else path / "steps.jsonl"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    events = read_log(find_log(sys.argv[1]))
    for e in events:
        if line := format_event(e):
            print(line)

    per_agent = defaultdict(lambda: {"tools": 0, "tokens": 0, "cost": 0.0})
    guard = []
    for e in events:
        if e["type"] == "tool":
            per_agent[e["agent"]]["tools"] += 1
            if e["status"] in ("blocked", "denied"):
                guard.append(e)
        elif e["type"] == "model":
            per_agent[e["agent"]]["tokens"] += e["tokens_in"] + e["tokens_out"]
            per_agent[e["agent"]]["cost"] += e["cost_usd"] or 0.0

    print("\nper agent:")
    for name, s in per_agent.items():
        print(f"  {name:<15} {s['tools']:>3} tool calls  {s['tokens']:>8,} tokens  ${s['cost']:.4f}")
    print(f"\nmost sub-agents running at once: {max_parallel(events)}")
    print(f"guardrail stops: {len(guard)}")
    for e in guard:
        print(f"  {e['status']:<8} {e['agent']}: {e['tool']} ({e.get('reason', '')})")


if __name__ == "__main__":
    main()
