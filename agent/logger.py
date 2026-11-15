"""Every step of a run goes to runs/<run_id>/steps.jsonl, one JSON object per line.

Event types:
    run_start       task, model, sandbox, limits
    plan            the orchestrator's decision for a round (delegate or finish)
    subagent_start  a sub-agent begins: role, instructions, output file
    model           one model response: tokens in/out, cost, which tools it called
    tool            one tool call: tool, input, output, status, duration
    subagent_end    the sub-agent's validated report
    check           the verdict on a finish attempt
    run_end         status and totals: steps, tokens, cost, seconds

Tool status is one of: ok, error, blocked (guardrail) or denied (human said no).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .sandbox import clip

MAX_FIELD = 2000
MARK = {"ok": "✓", "error": "✗", "blocked": "⛔ blocked", "denied": "✋ denied"}


def _clip(value):
    if isinstance(value, str):
        return clip(value, MAX_FIELD)
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


class RunLog:
    def __init__(self, path: Path, echo: bool = True):
        self.path = Path(path)
        self.echo = echo
        self.seq = 0
        self._t0 = time.monotonic()
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, type: str, **fields) -> dict:
        self.seq += 1
        event = {
            "seq": self.seq,
            "t": round(time.monotonic() - self._t0, 3),
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": type,
            **_clip(fields),
        }
        self._file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._file.flush()
        if self.echo and (line := format_event(event)):
            print(line, flush=True)
        return event

    def close(self) -> None:
        self._file.close()


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def short_input(tool: str, args: dict) -> str:
    if tool in ("read_file", "delete_file"):
        return args.get("path", "")
    if tool == "write_file":
        return f"{args.get('path', '')}, {len(args.get('content', '')):,} chars"
    if tool == "run_python":
        code = args.get("code", "").strip().splitlines() or [""]
        first = code[0][:48]
        return f"{first}{' …' if len(code) > 1 or len(code[0]) > 48 else ''} ({len(code)} lines)"
    if tool == "web_search":
        return repr(args.get("query", ""))
    if tool == "fetch_url":
        return args.get("url", "")
    return ""


def format_event(e: dict) -> str | None:
    """One readable line per event, used for the live view and by trace.py."""
    kind = e["type"]
    if kind == "run_start":
        return f"run {e['run_id']}  model={e['model']}  sandbox={e['sandbox']}  approval={e['approval']}"
    if kind == "plan":
        if e["action"] == "finish":
            return f"loop {e['round']:02d}  plan → finish: {', '.join(e['deliverables'])}"
        return f"loop {e['round']:02d}  plan → {len(e['subtasks'])} sub-agent(s): {e['thought'][:90]}"
    if kind == "subagent_start":
        return f"  ├─ {e['agent']} starts → {e['output_file']}"
    if kind == "tool":
        status = MARK.get(e["status"], e["status"])
        why = f" ({e['reason']})" if e.get("reason") else ""
        return f"  │    {e['agent']:<15} {e['tool']}({short_input(e['tool'], e['input'])}) {status}{why}"
    if kind == "subagent_end":
        return f"  ├─ {e['agent']} {e['status']}: {e['summary'][:100]}"
    if kind == "check":
        if e["done"]:
            return "  └─ check ✓ done"
        return "  └─ check ✗ " + "; ".join(e["problems"])[:140]
    if kind == "run_end":
        cost = f"${e['cost_usd']:.4f}"
        if e.get("unpriced_calls"):
            cost = f"≥{cost} ({e['unpriced_calls']} calls unpriced)" if e["cost_usd"] else "cost unknown"
        msg = f" ({e['message']})" if e.get("message") else ""
        return (f"{e['status']}{msg}: {e['steps']} steps · {e['tokens']:,} tokens · {cost} · "
                f"{e['seconds']:.0f} s")
    return None
