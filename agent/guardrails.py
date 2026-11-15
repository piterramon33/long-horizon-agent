"""Guardrails: what the agents may do, when a human must say yes, and when to stop.

1. Allowlist   each role gets a fixed set of tools; anything else is blocked
2. Paths       every path stays inside the workspace, and inputs/ is read-only
3. Approval    deleting files, or code that starts processes, uses the network,
               deletes files or reads environment variables, needs a human yes
4. Budget      hard caps on steps, tokens and dollars for the whole run

The sandbox is the real security boundary. These rules catch the obvious
mistakes early, ask before the risky ones, and leave a record in the log.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

from .sandbox import INPUTS

ROLE_TOOLS: dict[str, frozenset[str]] = {
    "researcher": frozenset({"list_files", "read_file", "write_file", "web_search", "fetch_url"}),
    "coder": frozenset({"list_files", "read_file", "write_file", "run_python", "delete_file"}),
    "writer": frozenset({"list_files", "read_file", "write_file"}),
}
WEB_TOOLS = frozenset({"web_search", "fetch_url"})
PATH_TOOLS = frozenset({"read_file", "write_file", "delete_file"})
WRITE_TOOLS = frozenset({"write_file", "delete_file"})

RISKY_CODE = [
    (re.compile(r"\bsubprocess\b|\bos\.(system|popen|exec\w*|spawn\w*|fork|kill)\b"), "starts or kills processes"),
    (re.compile(r"\bshutil\.(rmtree|move)\b|\bos\.(remove|unlink|rmdir|removedirs|rename|replace)\b"
                r"|\.unlink\(|\.rmdir\("), "deletes or moves files"),
    (re.compile(r"\b(socket|urllib|http\.client|requests|httpx|aiohttp|ftplib|smtplib)\b"), "uses the network"),
    (re.compile(r"\bctypes\b|\b__import__\b|\bimportlib\b"), "loads code dynamically"),
    (re.compile(r"\bos\.environ\b|\bgetenv\b"), "reads environment variables"),
]


class PolicyError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Verdict:
    action: str  # allow | ask | block
    reason: str = ""


def resolve_in_workspace(workspace: Path, path: str) -> Path:
    raw = (path or "").strip()
    norm = raw.replace("\\", "/")
    if not norm:
        raise PolicyError("empty path")
    if norm.startswith(("/", "~")) or PureWindowsPath(raw).drive:
        raise PolicyError(f"{raw!r} is an absolute path; use a path relative to the workspace")
    if ".." in PurePosixPath(norm).parts:
        raise PolicyError(f"{raw!r} points outside the workspace")
    root = Path(workspace).resolve()
    target = (root / norm).resolve()
    if not target.is_relative_to(root):  # e.g. through a symlink
        raise PolicyError(f"{raw!r} points outside the workspace")
    return target


def url_problem(url: str) -> str | None:
    """Why this URL may not be fetched, or None if it is fine."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "only http and https URLs can be fetched"
    host = (parsed.hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return f"{host or 'that'} is a local address"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return "private and local IP addresses are blocked"
    return None


class Policy:
    def __init__(self, workspace: Path, web: bool = True):
        self.workspace = Path(workspace).resolve()
        self.web = web

    def tools_for(self, role: str) -> frozenset[str]:
        tools = ROLE_TOOLS.get(role, frozenset())
        return tools if self.web else tools - WEB_TOOLS

    def check(self, role: str, tool: str, args: dict) -> Verdict:
        if tool not in self.tools_for(role):
            return Verdict("block", f"{tool} is not allowed for the {role} role")
        if tool in PATH_TOOLS:
            try:
                target = resolve_in_workspace(self.workspace, args.get("path", ""))
            except PolicyError as e:
                return Verdict("block", str(e))
            if tool in WRITE_TOOLS and target.is_relative_to(self.workspace / INPUTS):
                return Verdict("block", f"{INPUTS}/ holds the task inputs and is read-only")
            if tool == "delete_file":
                return Verdict("ask", f"delete {args.get('path')}")
        if tool == "run_python":
            reasons = [why for rx, why in RISKY_CODE if rx.search(args.get("code", ""))]
            if reasons:
                return Verdict("ask", "the code " + ", ".join(reasons))
        if tool == "fetch_url" and (problem := url_problem(args.get("url", ""))):
            return Verdict("block", problem)
        return Verdict("allow")


class Approver:
    """Human in the loop. ask: prompt in the terminal; deny: refuse; allow: never ask."""

    def __init__(self, mode: str = "ask"):
        if mode not in ("ask", "deny", "allow"):
            raise ValueError(f"APPROVAL must be ask, deny or allow, not {mode!r}")
        self.mode = mode
        self._lock: asyncio.Lock | None = None

    async def decide(self, agent: str, tool: str, reason: str, preview: str) -> bool:
        if self.mode == "allow":
            return True
        if self.mode == "deny" or not sys.stdin.isatty():  # nobody there to answer: refuse
            return False
        self._lock = self._lock or asyncio.Lock()
        async with self._lock:  # parallel sub-agents queue up for the one terminal
            body = "\n".join("      " + line for line in preview.splitlines()[:15])
            answer = await asyncio.to_thread(
                input, f"\n  ? {agent} wants to {tool} because {reason}:\n{body}\n    Allow? [y/N] "
            )
        return answer.strip().lower() in {"y", "yes", "s", "si", "sí"}


@dataclass
class Budget:
    max_steps: int
    max_tokens: int
    max_cost_usd: float
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def step(self) -> None:
        """Count one action (a tool call or an orchestrator decision) before it happens."""
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded(f"step limit reached ({self.max_steps})")

    def charge(self, tokens_in: int, tokens_out: int, cost: float | None) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += cost
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(f"token limit reached ({self.max_tokens:,})")
        if self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"cost limit reached (${self.max_cost_usd:.2f})")

    def left(self) -> str:
        return (f"{max(self.max_steps - self.steps, 0)} steps, "
                f"${max(self.max_cost_usd - self.cost_usd, 0):.3f}, "
                f"{max(self.max_tokens - self.tokens, 0):,} tokens")
