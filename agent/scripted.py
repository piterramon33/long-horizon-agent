"""A scripted stand-in for the LLM, used by `python run.py --demo` and the tests.

This is not an agent. It follows a fixed script for the demo task so you can
watch the loop, the parallel sub-agents, the sandbox, the guardrails and the
log without an API key. The coder tries two unsafe actions on purpose, so the
guardrails have something to stop.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic_ai import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

DEMO_INPUTS = Path(__file__).resolve().parent.parent / "eval_tasks" / "fixtures" / "cold_chain"
DEMO_TASK = (
    "inputs/ holds three cold-chain documents from different suppliers. Summarize each one "
    "(one researcher per document, in parallel), count how many temperature limits each document "
    "sets, and write brief.md with one section per document and a final Conflicts section."
)

NETWORK_CODE = 'import urllib.request\nprint(urllib.request.urlopen("https://example.com", timeout=5).status)\n'
STATS_CODE = '''\
import pathlib, re
rows = []
for p in sorted(pathlib.Path("notes").glob("*.md")):
    n = len(re.findall(r"-?\\d+\\s*°C", p.read_text(encoding="utf-8")))
    rows.append(f"| {p.stem} | {n} |")
table = "| document | temperature limits |\\n|---|---|\\n" + "\\n".join(rows)
pathlib.Path("stats.md").write_text(table + "\\n", encoding="utf-8")
print(table)
'''


def scripted_model() -> FunctionModel:
    return FunctionModel(_respond, model_name="scripted-demo")


# ---- reading the conversation so far ---------------------------------------
def _prompt(messages) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ""


def _tool_results(messages) -> list[tuple[str, dict, str]]:
    """(tool name, args, result) for every tool call so far, in order."""
    calls, results = {}, []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls[part.tool_call_id] = part
            elif isinstance(part, ToolReturnPart) and part.tool_call_id in calls:
                call = calls[part.tool_call_id]
                results.append((call.tool_name, call.args_as_dict(), str(part.content)))
    return results


def _call(*calls: tuple[str, dict]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls])


def _output(info: AgentInfo, kind: str, data: dict) -> ModelResponse:
    names = [t.name for t in info.output_tools]
    name = next((n for n in names if n.endswith(kind)), names[0])
    return _call((name, data))


# ---- the script ----------------------------------------------------------------
def _respond(messages, info: AgentInfo) -> ModelResponse:
    instructions = info.instructions or ""
    prompt, done = _prompt(messages), _tool_results(messages)
    if instructions.startswith("You are the orchestrator"):
        return _orchestrator(prompt, info)
    if instructions.startswith("You verify"):
        ok = "--- brief.md ---" in prompt and "## Conflicts" in prompt
        verdict = {"done": True} if ok else {"done": False, "problems": ["brief.md lacks a Conflicts section"]}
        return _output(info, "Check", verdict)
    if instructions.startswith("You are a researcher"):
        return _researcher(prompt, done, info)
    if instructions.startswith("You are a coder"):
        return _coder(done, info)
    if instructions.startswith("You are a writer"):
        return _writer(prompt, done, info)
    return ModelResponse(parts=[TextPart("I don't know what to do.")])


def _orchestrator(prompt: str, info: AgentInfo) -> ModelResponse:
    rnd = int(re.search(r"This is round (\d+)", prompt).group(1))
    docs = [f for f in re.findall(r"^(inputs/\S+)", prompt, re.M) if f.endswith((".md", ".txt"))][:3]
    if rnd == 1:
        subtasks = [{"role": "researcher", "output_file": f"notes/{Path(d).stem}.md",
                     "instructions": f"Read {d} and write notes with its temperature limits and its "
                                     "record-keeping and reporting rules."} for d in docs]
        return _output(info, "Delegate", {"thought": "One researcher per document; they are independent.",
                                          "subtasks": subtasks})
    if rnd == 2:
        return _output(info, "Delegate", {"thought": "Count the limits with code, not by eye.", "subtasks": [{
            "role": "coder", "output_file": "stats.md",
            "instructions": "Count how many temperature limits (values in °C) each file in notes/ mentions "
                            "and write the counts as a markdown table to stats.md."}]})
    if rnd == 3:
        return _output(info, "Delegate", {"thought": "Notes and counts are ready; write the brief.", "subtasks": [{
            "role": "writer", "output_file": "brief.md",
            "instructions": "Read every file in notes/ plus stats.md and write brief.md with one section "
                            "per document, the counts table, and a Conflicts section."}]})
    return _output(info, "Finish", {"summary": "Three documents summarized, limits counted, brief written.",
                                    "deliverables": ["brief.md"]})


def _researcher(prompt: str, done, info: AgentInfo) -> ModelResponse:
    doc = re.search(r"Read (\S+)", prompt).group(1)
    out = re.search(r"Write your result to: (\S+)", prompt).group(1)
    if not done:
        return _call(("read_file", {"path": doc}))
    if len(done) == 1:
        text = done[0][2]
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), doc)
        notes = f"# {title}\n\nSource: {doc}\n\n" + "\n".join(f"- {f}" for f in _facts(text)) + "\n"
        return _call(("write_file", {"path": out, "content": notes}))
    return _output(info, "SubResult", {"status": "done", "summary": f"Noted {doc}.", "files_written": [out]})


def _coder(done, info: AgentInfo) -> ModelResponse:
    script = [
        ("run_python", {"code": NETWORK_CODE}),                                    # needs approval
        ("write_file", {"path": "../outside.txt", "content": "escaped!"}),        # leaves the workspace
        ("run_python", {"code": STATS_CODE}),                                      # the real work
    ]
    if len(done) < len(script):
        return _call(script[len(done)])
    return _output(info, "SubResult", {"status": "done", "summary": "Counted limits per document. A network "
                                       "call and a write outside the workspace were stopped.",
                                       "files_written": ["stats.md"]})


def _writer(prompt: str, done, info: AgentInfo) -> ModelResponse:
    out = re.search(r"Write your result to: (\S+)", prompt).group(1)
    if not done:
        return _call(("list_files", {}))
    if len(done) == 1:
        wanted = [line.split()[0] for line in done[0][2].splitlines()
                  if line.startswith("notes/") or line.startswith("stats.md")]
        return _call(*[("read_file", {"path": p}) for p in wanted])  # several calls, run concurrently
    reads = {args["path"]: text for tool, args, text in done if tool == "read_file"}
    if not any(tool == "write_file" for tool, _, _ in done):
        return _call(("write_file", {"path": out, "content": _brief(reads)}))
    return _output(info, "SubResult", {"status": "done", "summary": f"Wrote {out}.", "files_written": [out]})


def _facts(text: str) -> list[str]:
    """Sentences (and table rows) that mention a temperature, records or reporting."""
    rows, prose = [], []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 2 and not set(cells[0]) <= set("-: "):
            rows.append(f"{cells[0]}: {cells[1]}")
        elif not line.startswith(("#", "|")):
            prose.append(line.strip(" -"))
    sentences = re.split(r"(?<=\.)\s+", " ".join(prose))
    return [f for f in rows + sentences if re.search(r"°C|record|retain|kept|report", f, re.I)]


TOPICS = {  # topic: (words that pick the sentence, the values to compare)
    "chilled range": (r"chilled", r"-?\d+\s*°C"),
    "record retention": (r"retain|kept", r"\d+\s*years?"),
    "reporting deadline": (r"report", r"\d+\s*hours?"),
}


def _brief(reads: dict[str, str]) -> str:
    notes = {Path(p).stem: t for p, t in reads.items() if p.startswith("notes/")}
    sections = []
    for text in notes.values():
        lines = text.strip().splitlines()
        sections.append(f"## {lines[0].lstrip('# ')}\n\n" + "\n".join(lines[1:]).strip())
    conflicts = []
    for topic, (pick, values) in TOPICS.items():
        found = {}
        for name, text in notes.items():
            for line in text.splitlines():
                if re.search(pick, line, re.I) and (hits := re.findall(values, line)):
                    found[name] = " to ".join(hits)
        if len(set(found.values())) > 1:
            conflicts.append(f"- **{topic}**: " + "; ".join(f"{k}: {v}" for k, v in found.items()))
    return ("# Cold-chain brief\n\n" + "\n\n".join(sections)
            + "\n\n## Temperature limits per document\n\n" + reads.get("stats.md", "").strip()
            + "\n\n## Conflicts\n\n" + ("\n".join(conflicts) or "- none found") + "\n")
