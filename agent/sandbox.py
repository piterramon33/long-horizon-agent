"""Where the agent's code runs.

The loop never runs model-written code in its own process. `run_python` goes to
one of these backends:

    docker  a throwaway container per call: no network, read-only root
            filesystem, no Linux capabilities, memory/CPU/process limits, and
            only the run's workspace mounted (inputs/ mounted read-only)
    e2b     a cloud microVM from e2b.dev with internet access turned off
    local   a plain subprocess in the workspace folder. NOT isolated: meant for
            the scripted demo and the unit tests

In every case the workspace folder on your disk is the source of truth, so
read_file and write_file behave the same whatever the backend.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

INPUTS = "inputs"


class SandboxUnavailable(RuntimeError):
    """The chosen backend cannot run here; the message says what to do."""


def clip(text: str, limit: int) -> str:
    """Shorten long text but keep both ends: the end of a traceback is the useful part."""
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 5, limit * 2 // 5
    return f"{text[:head]}\n... [{len(text) - head - tail:,} characters cut] ...\n{text[-tail:]}"


def _text(data) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else data


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def render(self, limit: int = 4000) -> str:
        lines = [f"exit code: {self.exit_code}" + (" (timed out)" if self.timed_out else "")]
        if self.stdout.strip():
            lines.append("stdout:\n" + clip(self.stdout, limit))
        if self.stderr.strip():
            lines.append("stderr:\n" + clip(self.stderr, limit))
        return "\n".join(lines)


class Sandbox:
    kind = "base"
    isolated = False

    def __init__(self, workspace: Path, timeout: int = 30):
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout

    def run_python(self, code: str) -> ExecResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DockerSandbox(Sandbox):
    kind = "docker"
    isolated = True

    def __init__(self, workspace: Path, timeout: int = 30, image: str = "python:3.12-slim",
                 memory: str = "512m", cpus: str = "1"):
        super().__init__(workspace, timeout)
        self.image, self.memory, self.cpus = image, memory, cpus

    def command(self, name: str) -> list[str]:
        cmd = [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "128",
            "-e", "HOME=/tmp", "-e", "PYTHONUTF8=1", "-e", "PYTHONIOENCODING=utf-8",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "--mount", f"type=bind,source={self.workspace},target=/workspace",
        ]
        inputs = self.workspace / INPUTS
        if inputs.is_dir():
            cmd += ["--mount", f"type=bind,source={inputs},target=/workspace/{INPUTS},readonly"]
        if hasattr(os, "getuid"):  # Linux/macOS: write files as you, not as root
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return cmd + ["-w", "/workspace", self.image, "python", "-"]

    def run_python(self, code: str) -> ExecResult:
        name = f"agent-sbx-{uuid.uuid4().hex[:10]}"
        try:
            p = subprocess.run(self.command(name), input=code, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=self.timeout)
            return ExecResult(p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return ExecResult(124, _text(e.stdout), _text(e.stderr) + f"\nKilled after {self.timeout} s.", True)

    @staticmethod
    def ensure_ready(image: str) -> None:
        if not shutil.which("docker"):
            raise SandboxUnavailable(
                "Docker is not installed. Install Docker Desktop, or set SANDBOX=e2b in .env."
            )
        info = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, text=True, timeout=30)
        if info.returncode != 0:
            raise SandboxUnavailable("Docker is installed but not running. Start Docker Desktop and try again.")
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
            print(f"Pulling {image} (first run only)...", flush=True)
            if subprocess.run(["docker", "pull", image]).returncode != 0:
                raise SandboxUnavailable(f"Could not pull {image}. Check your connection or DOCKER_IMAGE.")


class E2BSandbox(Sandbox):
    """One microVM per run. Files are synced up before and down after each code run."""

    kind = "e2b"
    isolated = True
    REMOTE = "/home/user/workspace"

    def __init__(self, workspace: Path, timeout: int = 30):
        super().__init__(workspace, timeout)
        try:
            from e2b_code_interpreter import Sandbox as E2B
        except ImportError as e:
            raise SandboxUnavailable("SANDBOX=e2b needs: pip install e2b-code-interpreter") from e
        if not os.environ.get("E2B_API_KEY"):
            raise SandboxUnavailable("SANDBOX=e2b needs E2B_API_KEY in .env (get one at e2b.dev).")
        self._sbx = E2B.create(allow_internet_access=False, timeout=3600)
        self._sent: dict[str, int] = {}

    def _push(self) -> None:
        for path in self.workspace.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self.workspace).as_posix()
                mtime = path.stat().st_mtime_ns
                if self._sent.get(rel) != mtime:
                    self._sbx.files.write(f"{self.REMOTE}/{rel}", path.read_bytes())
                    self._sent[rel] = mtime

    def _pull(self) -> None:
        for entry in self._sbx.files.list(self.REMOTE, depth=10):
            if getattr(entry.type, "value", entry.type) != "file":
                continue
            rel = entry.path[len(self.REMOTE) + 1:]
            if rel.split("/")[0] == INPUTS:  # inputs stay read-only on your disk
                continue
            target = self.workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._sbx.files.read(entry.path, format="bytes"))
            self._sent[rel] = target.stat().st_mtime_ns

    def run_python(self, code: str) -> ExecResult:
        self._push()
        ex = self._sbx.run_code(f"import os\nos.chdir({self.REMOTE!r})\n{code}", timeout=self.timeout)
        self._pull()
        stdout, stderr = "".join(ex.logs.stdout), "".join(ex.logs.stderr)
        if ex.error:
            return ExecResult(1, stdout, f"{stderr}\n{ex.error.name}: {ex.error.value}\n{ex.error.traceback}")
        return ExecResult(0, stdout, stderr)

    def close(self) -> None:
        self._sbx.kill()


class LocalSandbox(Sandbox):
    """A subprocess with a stripped environment (no API keys). It can still read
    and write anything your user can, so it is not a sandbox: demo and tests only."""

    kind = "local"
    isolated = False

    def run_python(self, code: str) -> ExecResult:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        try:
            p = subprocess.run([sys.executable, "-s", "-"], input=code, cwd=self.workspace, env=env,
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=self.timeout)
            return ExecResult(p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            return ExecResult(124, _text(e.stdout), _text(e.stderr) + f"\nKilled after {self.timeout} s.", True)


def make_sandbox(settings, workspace: Path) -> Sandbox:
    kind = settings.sandbox
    if kind == "docker":
        DockerSandbox.ensure_ready(settings.docker_image)
        return DockerSandbox(workspace, settings.exec_timeout, settings.docker_image)
    if kind == "e2b":
        return E2BSandbox(workspace, settings.exec_timeout)
    if kind == "local":
        if not settings.allow_local:
            raise SandboxUnavailable(
                "SANDBOX=local runs model-written code directly on your machine. Use docker or e2b, "
                "or pass --unsafe-local (or ALLOW_LOCAL_SANDBOX=1) if you accept that."
            )
        return LocalSandbox(workspace, settings.exec_timeout)
    raise SandboxUnavailable(f"Unknown SANDBOX={kind!r}; use docker, e2b or local.")
