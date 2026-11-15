"""The language model: any model on OpenRouter, or any OpenAI-compatible endpoint."""
from __future__ import annotations

from .config import Settings


def build_model(settings: Settings, name: str | None = None):
    name = name or settings.llm_model
    if not settings.llm_api_key or settings.llm_api_key.endswith("..."):
        raise RuntimeError(
            "LLM_API_KEY is not set. Copy .env.example to .env and paste your OpenRouter key, "
            "or try `python run.py --demo`, which needs no key."
        )
    if settings.llm_provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(api_key=settings.llm_api_key, app_title="long-horizon-agent")
        return OpenRouterModel(name, provider=provider)

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return OpenAIChatModel(name, provider=provider)


def model_label(model) -> str:
    return getattr(model, "model_name", None) or type(model).__name__


def response_cost(response, settings: Settings) -> float | None:
    """USD for one model response: the cost OpenRouter reports, else Pydantic AI's
    estimate from its price list, else PRICE_*_PER_M, else None (unknown)."""
    reported = (response.provider_details or {}).get("cost")
    if isinstance(reported, (int, float)):
        return float(reported)
    usage = response.usage
    if usage.cost is not None:
        return float(usage.cost)
    if settings.price_input_per_m is not None and settings.price_output_per_m is not None:
        return (usage.input_tokens * settings.price_input_per_m
                + usage.output_tokens * settings.price_output_per_m) / 1_000_000
    return None
