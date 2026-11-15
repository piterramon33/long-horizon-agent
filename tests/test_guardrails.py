import asyncio

import pytest
from pydantic import ValidationError

from agent.guardrails import Approver, Budget, BudgetExceeded, Policy, PolicyError, resolve_in_workspace, url_problem
from agent.schemas import Check, Delegate, SubTask


@pytest.mark.parametrize("path", ["../x.txt", "notes/../../x", "/etc/passwd", "C:\\Windows\\win.ini",
                                  "~/.ssh/id_rsa", "..\\..\\.env", ""])
def test_paths_outside_the_workspace_are_refused(tmp_path, path):
    with pytest.raises(PolicyError):
        resolve_in_workspace(tmp_path, path)


def test_relative_paths_resolve_inside(tmp_path):
    assert resolve_in_workspace(tmp_path, "notes/a.md") == (tmp_path / "notes" / "a.md").resolve()
    assert resolve_in_workspace(tmp_path, "notes\\a.md") == (tmp_path / "notes" / "a.md").resolve()


def test_role_allowlist(tmp_path):
    policy = Policy(tmp_path, web=True)
    assert policy.check("researcher", "run_python", {"code": "print(1)"}).action == "block"
    assert policy.check("writer", "delete_file", {"path": "a.md"}).action == "block"
    assert policy.check("coder", "web_search", {"query": "x"}).action == "block"
    assert policy.check("coder", "run_python", {"code": "print(sum(range(10)))"}).action == "allow"
    assert policy.check("researcher", "web_search", {"query": "x"}).action == "allow"
    assert Policy(tmp_path, web=False).check("researcher", "web_search", {"query": "x"}).action == "block"


def test_inputs_are_read_only(tmp_path):
    policy = Policy(tmp_path)
    assert policy.check("writer", "read_file", {"path": "inputs/a.md"}).action == "allow"
    assert policy.check("writer", "write_file", {"path": "inputs/a.md", "content": ""}).action == "block"
    assert policy.check("coder", "delete_file", {"path": "inputs/a.md"}).action == "block"


def test_deleting_needs_a_human(tmp_path):
    assert Policy(tmp_path).check("coder", "delete_file", {"path": "draft.md"}).action == "ask"


@pytest.mark.parametrize("code,why", [
    ("import subprocess; subprocess.run(['ls'])", "processes"),
    ("import os; os.system('rm -rf /')", "processes"),
    ("import shutil; shutil.rmtree('.')", "deletes"),
    ("import os\nos.remove('a.txt')", "deletes"),
    ("import urllib.request", "network"),
    ("import socket", "network"),
    ("import os; print(os.environ)", "environment"),
    ("__import__('o' + 's')", "dynamically"),
])
def test_risky_code_needs_a_human(tmp_path, code, why):
    verdict = Policy(tmp_path).check("coder", "run_python", {"code": code})
    assert verdict.action == "ask" and why in verdict.reason


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost:8000", "http://127.0.0.1/",
                                 "http://169.254.169.254/latest/meta-data", "http://10.0.0.5/", "ftp://x.org"])
def test_local_and_odd_urls_are_blocked(url):
    assert url_problem(url)


def test_public_urls_are_fine():
    assert url_problem("https://en.wikipedia.org/wiki/Cold_chain") is None


def test_budget_caps():
    b = Budget(max_steps=2, max_tokens=1000, max_cost_usd=0.01)
    b.step()
    b.step()
    with pytest.raises(BudgetExceeded, match="step"):
        b.step()
    with pytest.raises(BudgetExceeded, match="token"):
        Budget(10, 1000, 1).charge(900, 200, 0.0)
    with pytest.raises(BudgetExceeded, match="cost"):
        Budget(10, 10**6, 0.01).charge(10, 10, 0.02)
    unpriced = Budget(10, 10**6, 0.01)
    unpriced.charge(10, 10, None)
    assert unpriced.unpriced_calls == 1


def test_approver_modes():
    assert asyncio.run(Approver("allow").decide("a", "delete_file", "r", "p")) is True
    assert asyncio.run(Approver("deny").decide("a", "delete_file", "r", "p")) is False
    # "ask" with no terminal attached (pytest captures stdin) must refuse, not hang
    assert asyncio.run(Approver("ask").decide("a", "delete_file", "r", "p")) is False
    with pytest.raises(ValueError):
        Approver("maybe")


def test_schemas_fail_loudly():
    with pytest.raises(ValidationError):
        SubTask(role="coder", instructions="Do something useful here please", output_file="../escape.md")
    with pytest.raises(ValidationError):
        SubTask(role="hacker", instructions="Do something useful here please", output_file="a.md")
    task = {"role": "writer", "instructions": "Write the final report from notes/", "output_file": "a.md"}
    with pytest.raises(ValidationError, match="own output_file"):
        Delegate(thought="t", subtasks=[task, task])
    with pytest.raises(ValidationError):
        Delegate(thought="t", subtasks=[])
    with pytest.raises(ValidationError, match="problem"):
        Check(done=False)
