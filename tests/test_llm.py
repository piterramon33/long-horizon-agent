from decimal import Decimal

import pytest
from pydantic_ai import ModelResponse, RequestUsage, TextPart

from agent.config import Settings
from agent.llm import build_model, response_cost

PRICED = Settings(price_input_per_m=1.0, price_output_per_m=2.0)


def response(details=None, **usage):
    return ModelResponse(parts=[TextPart("x")], usage=RequestUsage(**usage), provider_details=details)


def test_cost_prefers_what_openrouter_reports():
    assert response_cost(response({"cost": 0.0012}, input_tokens=10, output_tokens=5), PRICED) == 0.0012


def test_cost_falls_back_to_estimate_then_env_prices_then_unknown():
    assert response_cost(response(input_tokens=1, output_tokens=1, cost=Decimal("0.5")), PRICED) == 0.5
    assert response_cost(response(input_tokens=1_000_000, output_tokens=1_000_000), PRICED) == 3.0
    assert response_cost(response(input_tokens=1, output_tokens=1), Settings()) is None


def test_missing_key_explains_what_to_do():
    with pytest.raises(RuntimeError, match="--demo"):
        build_model(Settings(llm_api_key="sk-or-..."))


def test_builds_an_openrouter_model_without_calling_it():
    model = build_model(Settings(llm_api_key="sk-or-test", llm_model="openai/gpt-5-mini"))
    assert model.model_name == "openai/gpt-5-mini"
