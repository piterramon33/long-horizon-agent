import os
import shutil
import subprocess

import pytest

from agent.config import Settings
from agent.sandbox import DockerSandbox, LocalSandbox, SandboxUnavailable, make_sandbox

IMAGE = os.environ.get("TEST_DOCKER_IMAGE", "python:3.12-slim")


def test_local_runs_code_in_the_workspace(tmp_path):
    result = LocalSandbox(tmp_path).run_python("open('out.txt', 'w').write('hi'); print('done')")
    assert result.exit_code == 0 and "done" in result.stdout
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_local_hides_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-or-secret")
    result = LocalSandbox(tmp_path).run_python("import os; print(os.environ.get('LLM_API_KEY'))")
    assert "sk-or-secret" not in result.stdout


def test_local_timeout(tmp_path):
    result = LocalSandbox(tmp_path, timeout=1).run_python("while True: pass")
    assert result.timed_out and result.exit_code == 124


def test_local_needs_explicit_opt_in(tmp_path):
    with pytest.raises(SandboxUnavailable, match="unsafe-local"):
        make_sandbox(Settings(sandbox="local", allow_local=False), tmp_path)


def test_docker_command_is_locked_down(tmp_path):
    (tmp_path / "inputs").mkdir()
    cmd = " ".join(DockerSandbox(tmp_path, image="img").command("box"))
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--pids-limit",
                 "--memory", "target=/workspace/inputs,readonly"):
        assert flag in cmd


def _docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        ok = subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
        return ok and subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _docker_ready(), reason=f"Docker not running or image {IMAGE} not pulled")
def test_docker_isolation(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "data.txt").write_text("keep me")
    box = DockerSandbox(tmp_path, timeout=20, image=IMAGE)
    code = """
import os, socket
def attempt(label, fn):
    try:
        fn(); print(label, "ALLOWED")
    except Exception as e:
        print(label, "BLOCKED", type(e).__name__)
attempt("network", lambda: socket.create_connection(("1.1.1.1", 53), timeout=3))
attempt("rootfs", lambda: open("/etc/evil", "w").write("x"))
attempt("inputs", lambda: open("inputs/data.txt", "w").write("gone"))
open("result.txt", "w").write("written in the box")
print("env", sorted(k for k in os.environ if "KEY" in k))
"""
    out = box.run_python(code)
    assert out.exit_code == 0, out.stderr
    assert "network BLOCKED" in out.stdout
    assert "rootfs BLOCKED" in out.stdout
    assert "inputs BLOCKED" in out.stdout
    assert "env []" in out.stdout
    assert (tmp_path / "result.txt").read_text() == "written in the box"
    assert (tmp_path / "inputs" / "data.txt").read_text() == "keep me"


@pytest.mark.skipif(not _docker_ready(), reason=f"Docker not running or image {IMAGE} not pulled")
def test_docker_timeout_kills_the_container(tmp_path):
    out = DockerSandbox(tmp_path, timeout=3, image=IMAGE).run_python("while True: pass")
    assert out.timed_out
    leftover = subprocess.run(["docker", "ps", "-q", "--filter", "name=agent-sbx-"], capture_output=True, text=True)
    assert leftover.stdout.strip() == ""
