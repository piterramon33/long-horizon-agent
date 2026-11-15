"""The loop. Start reading here.

    for round in range(MAX_ROUNDS):
        plan = orchestrator(task, memory)            # Delegate | Finish, validated
        if plan is Finish:
            verdict = check(task, plan)              # files exist? does a checker agree?
            if verdict.done: break
            memory.append(verdict.problems)
        else:
            results = sub_agents(plan.subtasks)      # in parallel; tools -> guardrails -> sandbox
            memory.append(results)
        # every model response and tool call is logged with tokens and cost

The budget (steps, tokens, dollars) can stop the run at any point.
"""
from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded

from .agents import CHECK_PROMPT, PLAN_PROMPT, checker_agent, orchestrator_agent, run_logged, worker_agent
from .config import Settings
from .guardrails import Approver, Budget, BudgetExceeded, Policy
from .llm import build_model, model_label
from .logger import RunLog
from .sandbox import INPUTS, SandboxUnavailable, clip, make_sandbox
from .schemas import Check, Delegate, Finish, SubResult, SubTask
from .tools import ToolBox

FATAL_HTTP = {401, 402, 403, 404}  # bad key, no credits, forbidden, unknown model: retrying won't help


def api_error(e: ModelHTTPError) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    detail = body.get("message") or str(e.body or "")
    return f"{e.model_name}: HTTP {e.status_code}: {detail[:300]}"


@dataclass
class RunResult:
    run_id: str
    status: str  # done | budget_exceeded | max_rounds | error
    message: str
    run_dir: Path
    steps: int
    tokens: int
    cost_usd: float
    seconds: float
    deliverables: list[str] = field(default_factory=list)
    fatal: bool = False  # the same error would hit every other run too (key, credits, sandbox)

    @property
    def workspace(self) -> Path:
        return self.run_dir / "workspace"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "steps.jsonl"


@dataclass
class Team:
    """Everything one run shares: model, agents, tools, log, budget."""

    model: object
    settings: Settings
    tools: ToolBox
    log: RunLog
    budget: Budget

    def __post_init__(self):
        self.orchestrator = orchestrator_agent(self.model, self.settings.web)
        self.checker = checker_agent(self.model)

    async def _run(self, agent, prompt: str, name: str, request_limit: int):
        return await run_logged(agent, prompt, name=name, log=self.log, budget=self.budget,
                                settings=self.settings, request_limit=request_limit)

    # ---- plan -----------------------------------------------------------
    async def plan(self, task: str, memory: list[str], rnd: int) -> Delegate | Finish:
        prompt = PLAN_PROMPT.format(
            task=task, files=self.tools.list_files(),
            memory="\n\n".join(memory) or "(nothing yet: this is the first round)",
            round=rnd, max_rounds=self.settings.max_rounds, budget=self.budget.left(),
        )
        plan = await self._run(self.orchestrator, prompt, "orchestrator", request_limit=4)
        self.budget.step()
        action = "finish" if isinstance(plan, Finish) else "delegate"
        self.log.write("plan", agent="orchestrator", round=rnd, action=action, **plan.model_dump())
        return plan

    # ---- act ------------------------------------------------------------
    async def delegate(self, plan: Delegate, rnd: int) -> list[tuple[str, SubResult]]:
        gate = asyncio.Semaphore(self.settings.max_parallel)
        jobs = [self._sub_agent(f"{st.role}-{rnd}.{i}", st, gate) for i, st in enumerate(plan.subtasks, 1)]
        results = await asyncio.gather(*jobs, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):  # only BudgetExceeded gets this far
                raise r
        return results

    async def _sub_agent(self, name: str, st: SubTask, gate: asyncio.Semaphore) -> tuple[str, SubResult]:
        async with gate:
            self.log.write("subagent_start", agent=name, role=st.role, instructions=st.instructions,
                           output_file=st.output_file)
            worker = worker_agent(self.model, st.role, self.tools.tools_for(name, st.role), self.settings.web)
            prompt = f"{st.instructions}\n\nWrite your result to: {st.output_file}\nThen return your report."
            try:
                result = await self._run(worker, prompt, name, self.settings.subagent_max_requests)
            except BudgetExceeded:
                raise
            except ModelHTTPError as e:
                if e.status_code in FATAL_HTTP:
                    raise
                result = SubResult(status="failed", summary=api_error(e))
            except UsageLimitExceeded:
                result = SubResult(status="failed", summary=f"stopped after "
                                   f"{self.settings.subagent_max_requests} model calls without finishing")
            except UnexpectedModelBehavior as e:
                result = SubResult(status="failed", summary=f"gave invalid output repeatedly: {e}")
            except Exception as e:
                result = SubResult(status="failed", summary=f"{type(e).__name__}: {e}")
            if result.status == "done" and not (self.tools.workspace / st.output_file).is_file():
                result = result.model_copy(update={"status": "failed",
                                                   "summary": f"{result.summary} (but {st.output_file} was not written)"})
            self.log.write("subagent_end", agent=name, **result.model_dump())
            return name, result

    # ---- check ----------------------------------------------------------
    async def check(self, task: str, plan: Finish, rnd: int) -> Check:
        ws = self.tools.workspace
        missing = [p for p in plan.deliverables if not (ws / p).is_file() or (ws / p).stat().st_size == 0]
        if missing:  # cheap, deterministic check first: no model call needed
            verdict = Check(done=False, problems=[f"{p} does not exist or is empty" for p in missing])
        else:
            contents = "\n\n".join(
                f"--- {p} ---\n{clip((ws / p).read_text(encoding='utf-8', errors='replace'), 6000)}"
                for p in plan.deliverables
            )
            prompt = CHECK_PROMPT.format(task=task, summary=plan.summary, files=self.tools.list_files(),
                                         contents=contents)
            verdict = await self._run(self.checker, prompt, "checker", request_limit=4)
            self.budget.step()
        self.log.write("check", agent="checker", round=rnd, **verdict.model_dump())
        return verdict


def round_summary(rnd: int, results: list[tuple[str, SubResult]]) -> str:
    lines = [f"Round {rnd}:"]
    for name, r in results:
        files = ", ".join(r.files_written) or "none"
        lines.append(f"- {name} [{r.status}] {r.summary} Files written: {files}")
    return "\n".join(lines)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def copy_inputs(inputs, dest: Path) -> list[str]:
    """Files are copied into inputs/; a folder's contents are copied into inputs/."""
    copied = []
    for item in inputs:
        src = Path(item)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        elif src.is_file():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src.name)
        else:
            raise FileNotFoundError(f"input not found: {src}")
    if dest.is_dir():
        copied = sorted(p.relative_to(dest.parent).as_posix() for p in dest.rglob("*") if p.is_file())
    return copied


async def run_task(task: str, settings: Settings, *, inputs=(), model=None, echo: bool = True,
                   run_id: str | None = None) -> RunResult:
    run_id = run_id or new_run_id()
    run_dir = Path(settings.runs_dir) / run_id
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    copied = copy_inputs(inputs, workspace / INPUTS)

    log = RunLog(run_dir / "steps.jsonl", echo=echo)
    budget = Budget(settings.max_steps, settings.max_tokens, settings.max_cost_usd)
    status, message, deliverables, sandbox, fatal = "max_rounds", "", [], None, False
    t0 = time.monotonic()
    try:
        model = model or build_model(settings)
        log.write("run_start", run_id=run_id, task=task, model=model_label(model), sandbox=settings.sandbox,
                  approval=settings.approval, web=settings.web, inputs=copied,
                  limits={"rounds": settings.max_rounds, "steps": settings.max_steps,
                          "tokens": settings.max_tokens, "cost_usd": settings.max_cost_usd})
        sandbox = make_sandbox(settings, workspace)
        tools = ToolBox(workspace, sandbox, Policy(workspace, settings.web), Approver(settings.approval),
                        budget, log)
        team = Team(model, settings, tools, log, budget)

        memory: list[str] = []
        for rnd in range(1, settings.max_rounds + 1):
            plan = await team.plan(task, memory, rnd)                       # plan
            if isinstance(plan, Finish):
                verdict = await team.check(task, plan, rnd)                 # check
                if verdict.done:
                    status, deliverables = "done", plan.deliverables
                    break
                memory.append(f"Round {rnd}: you chose Finish but the check failed: "
                              + "; ".join(verdict.problems))
            else:
                results = await team.delegate(plan, rnd)                    # act
                memory.append(round_summary(rnd, results))
        else:
            message = f"no finish after {settings.max_rounds} rounds"
    except BudgetExceeded as e:
        status, message = "budget_exceeded", str(e)
    except ModelHTTPError as e:
        status, message, fatal = "error", api_error(e), e.status_code in FATAL_HTTP
    except UnexpectedModelBehavior as e:  # outputs kept failing validation: fail loudly
        status, message = "error", f"invalid model output: {e}"
    except SandboxUnavailable as e:
        status, message, fatal = "error", str(e), True
    except RuntimeError as e:
        status, message = "error", str(e)
    finally:
        if sandbox:
            sandbox.close()
        seconds = time.monotonic() - t0
        log.write("run_end", status=status, message=message, steps=budget.steps, tokens=budget.tokens,
                  tokens_in=budget.tokens_in, tokens_out=budget.tokens_out, cost_usd=round(budget.cost_usd, 6),
                  unpriced_calls=budget.unpriced_calls, seconds=round(seconds, 1), deliverables=deliverables)
        log.close()
    return RunResult(run_id, status, message, run_dir, budget.steps, budget.tokens, budget.cost_usd,
                     seconds, deliverables, fatal)
