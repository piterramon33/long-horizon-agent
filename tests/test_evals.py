"""The test set itself must be right: tasks load, and the checks accept correct answers."""
import csv
import re
from collections import defaultdict
from pathlib import Path

import pytest

from agent.evals import TASKS_DIR, load_tasks, run_check
from agent.loop import RunResult

KNOWN = {"file_exists", "file_contains", "file_not_contains", "inputs_intact", "tool_never_succeeded", "tool_never_called",
         "min_steps", "min_parallel", "sandbox_python"}


def checks_of(task_id: str) -> list[dict]:
    return next(t for t in load_tasks() if t["id"] == task_id)["checks"]


def patterns(task_id: str, path: str) -> list[str]:
    return [c["pattern"] for c in checks_of(task_id) if c["type"] == "file_contains" and c["path"] == path]


def test_tasks_load_and_inputs_exist():
    tasks = load_tasks()
    assert len(tasks) == 5 and len({t["id"] for t in tasks}) == 5
    for t in tasks:
        assert all(Path(p).exists() for p in t["inputs"]), t["id"]
        assert {c["type"] for c in t["checks"]} <= KNOWN, t["id"]


@pytest.mark.parametrize("answer", ["There are 1229 primes below 10,000.\nThey add up to 5736396.",
                                    "Count: 1,229 primes. Sum: 5,736,396."])
def test_primes_checks_accept_the_right_answer(answer):
    assert all(re.search(p, answer, re.I | re.M) for p in patterns("primes", "answer.md"))


def test_sales_checks_match_the_real_totals():
    by_region, by_month = defaultdict(float), defaultdict(float)
    with open(TASKS_DIR / "fixtures" / "sales" / "sales.csv", newline="") as f:
        for row in csv.DictReader(f):
            revenue = int(row["units"]) * float(row["unit_price"])
            by_region[row["region"]] += revenue
            by_month[row["date"][:7]] += revenue
    best_month, best = max(by_month.items(), key=lambda kv: kv[1])
    for fmt in ("{:,.2f}", "{:.2f}", "${:,.0f}"):
        table = "\n".join(f"| {r} | {fmt.format(v)} |" for r, v in sorted(by_region.items(), key=lambda kv: -kv[1]))
        report = f"{table}\n\nBest month: {best_month} with {fmt.format(best)}"
        assert all(re.search(p, report, re.I | re.M) for p in patterns("sales_report", "report.md")), fmt


def test_cold_chain_patterns():
    five = next(p for p in patterns("cold_chain_brief", "brief.md") if "5" in p and "°" in p)
    for good in ("0 °C to 5 °C", "0-5°C", "up to 5 ºC"):
        assert re.search(five, good), good
    for bad in ("-15 °C", "25 °C", "1.5 °C"):
        assert not re.search(five, bad), bad


def fake_run(tmp_path, steps=12) -> RunResult:
    (tmp_path / "workspace" / "inputs").mkdir(parents=True)
    return RunResult("t", "done", "", tmp_path, steps, 100, 0.0, 1.0)


def test_run_check_types(tmp_path, settings):
    result = fake_run(tmp_path)
    ws = result.workspace
    (ws / "report.md").write_text("Total: 42")
    src = tmp_path / "fixture"
    src.mkdir()
    (src / "a.txt").write_text("original")
    (ws / "inputs" / "a.txt").write_text("original")
    task = {"inputs": [src]}
    events = [{"type": "subagent_start"}, {"type": "subagent_start"}, {"type": "subagent_end"},
              {"type": "tool", "tool": "delete_file", "status": "denied"}]

    def check(c):
        return run_check(c, task, result, events, settings)[0]

    assert check({"type": "file_exists", "path": "report.md"})
    assert check({"type": "file_contains", "path": "report.md", "pattern": "total: 4\\d"})
    assert not check({"type": "file_not_contains", "path": "report.md", "pattern": "42"})
    assert check({"type": "inputs_intact"})
    assert check({"type": "tool_never_succeeded", "tool": "delete_file"})
    assert not check({"type": "tool_never_called", "tool": "delete_file"})
    assert check({"type": "tool_never_called", "tool": "run_python"})
    assert check({"type": "min_steps", "n": 12}) and not check({"type": "min_steps", "n": 13})
    assert check({"type": "min_parallel", "n": 2}) and not check({"type": "min_parallel", "n": 3})
    assert check({"type": "sandbox_python", "code": "assert open('report.md').read().endswith('42')"})
    assert not check({"type": "sandbox_python", "code": "raise SystemExit(1)"})
    (ws / "inputs" / "a.txt").write_text("tampered")
    assert not check({"type": "inputs_intact"})


def test_stats_hidden_tests_accept_a_correct_module(tmp_path, settings):
    result = fake_run(tmp_path)
    (result.workspace / "stats.py").write_text(CORRECT_STATS)
    (result.workspace / "test_stats.py").write_text(TESTS)
    (result.workspace / "test_edge.py").write_text(TESTS.replace("T(", "E("))
    for c in checks_of("stats_module"):
        if c["type"] == "sandbox_python":
            ok, detail = run_check(c, {"inputs": []}, result, [], settings)
            assert ok, detail


CORRECT_STATS = """
def _check(xs):
    if not xs:
        raise ValueError("empty")

def mean(xs):
    _check(xs)
    return sum(xs) / len(xs)

def median(xs):
    _check(xs)
    s, n = sorted(xs), len(xs)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def mode(xs):
    _check(xs)
    counts = {x: xs.count(x) for x in xs}
    top = max(counts.values())
    return min(x for x, c in counts.items() if c == top)

def stdev(xs):
    _check(xs)
    if len(xs) < 2:
        raise ValueError("need two points")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
"""

TESTS = """
import unittest, stats

class T(unittest.TestCase):
    def test_mean(self): self.assertEqual(stats.mean([1, 2, 3]), 2)
    def test_median(self): self.assertEqual(stats.median([1, 3, 2]), 2)
    def test_mode(self): self.assertEqual(stats.mode([1, 1, 2]), 1)
    def test_stdev(self): self.assertAlmostEqual(stats.stdev([1, 3]), 2 ** 0.5)
"""
