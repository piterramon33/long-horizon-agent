import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from agent.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """Offline settings: local sandbox, nobody to approve, no web."""
    return Settings(runs_dir=tmp_path / "runs", sandbox="local", allow_local=True, approval="deny", web=False)
