"""Settings, read from the environment (.env is loaded if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _float_or_none(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def _bool(name: str, default: bool) -> bool:
    return _str(name, "on" if default else "off").lower() in {"1", "on", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # model
    llm_provider: str = "openrouter"  # openrouter | openai (any OpenAI-compatible endpoint)
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-5-mini"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    price_input_per_m: float | None = None  # USD per 1M tokens, if the endpoint reports no cost
    price_output_per_m: float | None = None
    max_output_tokens: int = 8192  # per model response; OpenRouter reserves credit for this much

    # sandbox
    sandbox: str = "docker"  # docker | e2b | local
    docker_image: str = "python:3.12-slim"
    exec_timeout: int = 30
    allow_local: bool = False  # local runs model-written code unsandboxed

    # guardrails
    approval: str = "ask"  # ask | deny | allow
    web: bool = True
    max_rounds: int = 8
    max_steps: int = 60
    max_tokens: int = 500_000
    max_cost_usd: float = 0.50
    subagent_max_requests: int = 15
    max_parallel: int = 3

    runs_dir: Path = Path("runs")

    def with_(self, **changes) -> "Settings":
        """Copy with some fields changed; None values are ignored (handy for CLI flags)."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})


def load_settings(**overrides) -> Settings:
    settings = Settings(
        llm_provider=_str("LLM_PROVIDER", "openrouter"),
        llm_api_key=_str("LLM_API_KEY", ""),
        llm_model=_str("LLM_MODEL", Settings.llm_model),
        llm_base_url=_str("LLM_BASE_URL", Settings.llm_base_url),
        price_input_per_m=_float_or_none("PRICE_INPUT_PER_M"),
        price_output_per_m=_float_or_none("PRICE_OUTPUT_PER_M"),
        max_output_tokens=_int("MAX_OUTPUT_TOKENS", 8192),
        sandbox=_str("SANDBOX", "docker"),
        docker_image=_str("DOCKER_IMAGE", Settings.docker_image),
        exec_timeout=_int("EXEC_TIMEOUT", 30),
        allow_local=_bool("ALLOW_LOCAL_SANDBOX", False),
        approval=_str("APPROVAL", "ask"),
        web=_bool("WEB", True),
        max_rounds=_int("MAX_ROUNDS", 8),
        max_steps=_int("MAX_STEPS", 60),
        max_tokens=_int("MAX_TOKENS", 500_000),
        max_cost_usd=_float("MAX_COST_USD", 0.50),
        subagent_max_requests=_int("SUBAGENT_MAX_REQUESTS", 15),
        max_parallel=_int("MAX_PARALLEL", 3),
        runs_dir=Path(_str("RUNS_DIR", "runs")),
    )
    return settings.with_(**overrides)
