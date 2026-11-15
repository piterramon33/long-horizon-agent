"""Run the task test set and print a pass rate, per model, so models can be compared.

    python eval.py                                   # every task, model from .env
    python eval.py --models openai/gpt-5-mini,google/gemini-3.7-flash
    python eval.py --only primes,stats_module -v     # some tasks, with the live trace

Approval is forced to "deny": nobody is at the keyboard during an eval, so any
risky action is refused (and shows up in the log). Results are saved to
runs/eval-<timestamp>/results.json, and every task keeps its own run folder.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from agent.config import load_settings
from agent.evals import load_tasks, run_check
from agent.llm import build_model
from agent.logger import read_log
from agent.loop import run_task

sys.stdout.reconfigure(errors="replace")


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in name)


async def evaluate(model_name: str, model, tasks: list[dict], settings, out_dir: Path, verbose: bool) -> list[dict]:
    rows = []
    for task in tasks:
        task_settings = settings.with_(runs_dir=out_dir / slug(model_name))
        result = await run_task(task["prompt"], task_settings, inputs=task["inputs"], model=model,
                                echo=verbose, run_id=task["id"])
        events = read_log(result.log_path)
        checks = [(c["type"], *run_check(c, task, result, events, task_settings)) for c in task["checks"]]
        passed = result.status == "done" and all(ok for _, ok, _ in checks)
        print(f"  {'PASS' if passed else 'FAIL'}  {task['id']:<18} {result.steps:>3} steps "
              f"{result.tokens:>9,} tok  ${result.cost_usd:.4f}  {result.seconds:>4.0f} s  [{result.status}]",
              flush=True)
        if result.status == "error":  # the checks fail because of this; no need to list them
            print(f"        {result.message[:200]}")
        elif not passed:
            if result.message:
                print(f"        {result.message[:200]}")
            for detail in dict.fromkeys(d for _, ok, d in checks if not ok):
                print(f"        ✗ {detail}")
        rows.append({"model": model_name, "task": task["id"], "passed": passed, "status": result.status,
                     "steps": result.steps, "tokens": result.tokens, "cost_usd": round(result.cost_usd, 6),
                     "seconds": round(result.seconds, 1), "run_dir": str(result.run_dir),
                     "checks": [{"type": k, "ok": ok, "detail": d} for k, ok, d in checks]})
        if result.fatal:
            print(f"  skipping the other tasks for {model_name}: this error would repeat")
            break
    return rows


async def main_async(args) -> int:
    settings = load_settings(sandbox=args.sandbox, approval="deny")
    if args.demo:
        settings = settings.with_(sandbox=args.sandbox or "local", allow_local=True, web=False)
    models = args.models.split(",") if args.models else [settings.llm_model]
    only = args.only.split(",") if args.only else (["cold_chain_brief"] if args.demo else None)
    tasks = load_tasks(only=only)
    if not tasks:
        print("no tasks matched")
        return 1
    out_dir = Path(settings.runs_dir) / f"eval-{datetime.now():%Y%m%d-%H%M%S}"

    all_rows = []
    for name in models:
        if args.demo:
            from agent.scripted import scripted_model

            name, model = "scripted-demo", scripted_model()
            print("Demo: a scripted model that only knows the cold-chain task. This tests the plumbing, not an agent.")
        else:
            try:
                model = build_model(settings, name)
            except RuntimeError as e:
                print(e)
                return 1
        print(f"\n{name}  ({len(tasks)} tasks, sandbox={settings.sandbox})")
        all_rows += await evaluate(name, model, tasks, settings, out_dir, args.verbose)

    print(f"\n{'model':<32} {'pass rate':>10} {'avg steps':>10} {'tokens':>10} {'cost':>9} {'time':>7}")
    exit_code = 0
    for name in dict.fromkeys(r["model"] for r in all_rows):
        rows = [r for r in all_rows if r["model"] == name]
        rate = sum(r["passed"] for r in rows) / len(tasks)
        print(f"{name:<32} {rate:>9.0%} {sum(r['steps'] for r in rows) / len(rows):>10.1f} "
              f"{sum(r['tokens'] for r in rows):>10,} ${sum(r['cost_usd'] for r in rows):>8.4f} "
              f"{sum(r['seconds'] for r in rows):>6.0f}s   ({sum(r['passed'] for r in rows)}/{len(tasks)})")
        exit_code |= rate < args.min
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir / 'results.json'}")
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated model names (default: LLM_MODEL)")
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--sandbox", choices=["docker", "e2b", "local"])
    ap.add_argument("--min", type=float, default=0.0, help="exit 1 if a model's pass rate is below this (0-1)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the live trace of every run")
    ap.add_argument("--demo", action="store_true", help="scripted model, to test the plumbing without a key")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
