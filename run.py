"""Run the agent on a task.

    python run.py "Research X and write report.md"
    python run.py "Analyze the attached data and write report.md" --input data.csv
    python run.py --demo          # scripted model: no API key, no Docker

Everything the run does is logged to runs/<run_id>/steps.jsonl and the files
it produces are in runs/<run_id>/workspace/. View a run with trace.py.
"""
import argparse
import asyncio
import sys

from agent.config import load_settings
from agent.llm import build_model
from agent.loop import run_task

sys.stdout.reconfigure(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="what the agent should do, in plain words")
    ap.add_argument("--input", action="append", default=[], help="file or folder to copy into inputs/ (repeatable)")
    ap.add_argument("--model", help="e.g. openai/gpt-5-mini (default: LLM_MODEL from .env)")
    ap.add_argument("--sandbox", choices=["docker", "e2b", "local"])
    ap.add_argument("--approval", choices=["ask", "deny", "allow"])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-cost", type=float, help="USD")
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--no-web", action="store_true", help="disable web_search and fetch_url")
    ap.add_argument("--unsafe-local", action="store_true", help="allow SANDBOX=local (no isolation)")
    ap.add_argument("--quiet", action="store_true", help="no live trace")
    ap.add_argument("--demo", action="store_true", help="scripted model on a demo task, no API key needed")
    args = ap.parse_args()

    settings = load_settings(llm_model=args.model, sandbox=args.sandbox, approval=args.approval,
                             max_steps=args.max_steps, max_cost_usd=args.max_cost, max_rounds=args.max_rounds,
                             web=False if args.no_web else None, allow_local=True if args.unsafe_local else None)
    model, task, inputs = None, args.task, args.input
    if args.demo:
        from agent.scripted import DEMO_INPUTS, DEMO_TASK, scripted_model

        model, task, inputs = scripted_model(), DEMO_TASK, [DEMO_INPUTS]
        settings = settings.with_(sandbox=args.sandbox or "local", allow_local=True, web=False,
                                  approval=args.approval or "deny")
        print("Demo: a scripted model stands in for the LLM (see agent/scripted.py).\n")
    elif not task:
        ap.error("give a task, or use --demo")
    else:
        try:
            model = build_model(settings)
        except RuntimeError as e:
            print(e)
            return 1

    result = asyncio.run(run_task(task, settings, inputs=inputs, model=model, echo=not args.quiet))

    print()
    if args.quiet:
        why = f" ({result.message})" if result.message else ""
        print(f"{result.status}{why}: {result.steps} steps · {result.tokens:,} tokens · "
              f"${result.cost_usd:.4f} · {result.seconds:.0f} s")
    for path in result.deliverables:
        print(f"deliverable: {result.workspace / path}")
    print(f"workspace:   {result.workspace}")
    print(f"log:         {result.log_path}")
    print(f"view it:     python trace.py {result.run_dir}")
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
