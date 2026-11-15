"""End-to-end runs of the loop with a scripted model (no API key, no network)."""
import asyncio

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agent.evals import max_parallel
from agent.logger import read_log
from agent.loop import run_task
from agent.scripted import DEMO_INPUTS, DEMO_TASK, scripted_model


def run_demo(settings):
    return asyncio.run(run_task(DEMO_TASK, settings, inputs=[DEMO_INPUTS], model=scripted_model(), echo=False))


def test_demo_runs_long_parallel_logged_and_guarded(settings):
    result = run_demo(settings)
    events = read_log(result.log_path)

    assert result.status == "done", result.message
    assert result.steps >= 10                                   # runs 10+ steps on its own
    assert max_parallel(events) == 3                            # three researchers at once
    brief = (result.workspace / "brief.md").read_text(encoding="utf-8")
    assert "## Conflicts" in brief and "0 °C to 5 °C" in brief

    tools = [e for e in events if e["type"] == "tool"]
    assert all({"agent", "tool", "input", "output", "status", "ms"} <= e.keys() for e in tools)
    models = [e for e in events if e["type"] == "model"]
    assert models and all("tokens_in" in e and "tokens_out" in e and "cost_usd" in e for e in models)
    assert [e["type"] for e in events][0] == "run_start" and events[-1]["type"] == "run_end"

    statuses = {(e["tool"], e["status"]) for e in tools}
    assert ("run_python", "denied") in statuses                 # network code needed a human
    assert ("write_file", "blocked") in statuses                # ../outside.txt
    assert not (result.run_dir / "outside.txt").exists()


def test_step_cap_stops_the_run(settings):
    result = run_demo(settings.with_(max_steps=5))
    assert result.status == "budget_exceeded" and "step limit" in result.message
    assert read_log(result.log_path)[-1]["status"] == "budget_exceeded"


def test_token_cap_stops_the_run(settings):
    result = run_demo(settings.with_(max_tokens=1500))
    assert result.status == "budget_exceeded" and "token limit" in result.message


def test_round_cap_stops_the_run(settings):
    result = run_demo(settings.with_(max_rounds=2))
    assert result.status == "max_rounds"


def test_invalid_output_fails_loudly(settings):
    def always_invalid(messages, info):  # a Delegate with no subtasks never validates
        name = next(t.name for t in info.output_tools if t.name.endswith("Delegate"))
        return ModelResponse(parts=[ToolCallPart(name, {"thought": "hmm", "subtasks": []})])

    result = asyncio.run(run_task("anything", settings, model=FunctionModel(always_invalid), echo=False))
    assert result.status == "error" and "invalid model output" in result.message


def test_missing_deliverable_fails_the_check_without_a_model_call(settings):
    calls = {"n": 0}

    def finish_immediately(messages, info):
        calls["n"] += 1
        name = next(t.name for t in info.output_tools if t.name.endswith("Finish"))
        return ModelResponse(parts=[ToolCallPart(name, {"summary": "done!", "deliverables": ["report.md"]})])

    result = asyncio.run(run_task("write report.md", settings.with_(max_rounds=2),
                                  model=FunctionModel(finish_immediately), echo=False))
    checks = [e for e in read_log(result.log_path) if e["type"] == "check"]
    assert result.status == "max_rounds"
    assert checks and not checks[0]["done"] and "report.md" in checks[0]["problems"][0]
    assert calls["n"] == 2  # only the orchestrator was called; the checker never ran


def test_every_request_caps_its_output_tokens(settings):
    seen = []

    def spy(messages, info):
        seen.append((info.model_settings or {}).get("max_tokens"))
        name = next(t.name for t in info.output_tools if t.name.endswith("Finish"))
        return ModelResponse(parts=[ToolCallPart(name, {"summary": "s", "deliverables": ["x.md"]})])

    asyncio.run(run_task("t", settings.with_(max_rounds=1, max_output_tokens=1234), model=FunctionModel(spy),
                         echo=False))
    assert seen == [1234]


def test_out_of_credits_is_short_and_fatal(settings):
    from pydantic_ai import ModelHTTPError

    def broke(messages, info):
        raise ModelHTTPError(402, "openai/gpt-5-mini", {"message": "This request requires more credits"})

    result = asyncio.run(run_task("t", settings, model=FunctionModel(broke), echo=False))
    assert result.status == "error" and result.fatal
    assert result.message == "openai/gpt-5-mini: HTTP 402: This request requires more credits"
