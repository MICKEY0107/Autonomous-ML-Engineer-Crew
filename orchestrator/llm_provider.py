"""Configured CrewAI LLM providers for agentic pipeline decisions.

Provider chain (tried in order, each only once - no retry storms):

1. Primary:   ``LLM_PROVIDER`` + ``CREWAI_MODEL`` (e.g. gemini/gemini-3.5-flash-lite)
2. Fallbacks: ``LLM_FALLBACK_CHAIN`` - comma-separated model strings such as
              ``ollama/qwen2.5-coder:latest,openrouter/google/gemini-3.5-flash-lite``.
              When unset, the other cloud provider is derived as before.
3. If every provider fails, callers use their typed deterministic policy.

The provider is read from the model prefix: ``gemini/``, ``openrouter/`` or
``ollama/`` (a local model served by Ollama - no API cost, data stays on the
machine).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from crewai import LLM


load_dotenv()

LOCAL_PROVIDERS = frozenset({"ollama"})


def provider_of(model: str, default: str = "") -> str:
    """Provider name from a model string's prefix."""

    prefix = model.split("/", 1)[0].lower() if "/" in model else ""
    return prefix if prefix in {"gemini", "openrouter", "ollama"} else default.lower()


def _build_llm(provider: str, model: str) -> "LLM":
    from crewai import LLM

    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "700"))
    timeout_s = int(os.getenv("LLM_FALLBACK_TIMEOUT_SECONDS", "15"))
    if provider == "gemini":
        from google.genai import types

        extra: dict = {}
        # Optional: only some Gemini models accept a thinking budget (newer ones
        # reject thinking_budget=0 with 400 INVALID_ARGUMENT), so it is opt-in.
        budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
        if budget:
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=int(budget))
        return LLM(
            model=model,
            api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.1,
            max_output_tokens=max_tokens,
            client_params={
                "http_options": types.HttpOptions(
                    timeout=timeout_s * 1_000,
                    # Fail fast: one attempt, then the next provider in the chain.
                    retry_options=types.HttpRetryOptions(attempts=1),
                )
            },
            **extra,
        )
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.environ[
            "OPEN_ROUTER_API_KEY"
        ]
        return LLM(
            model=model,
            api_key=api_key,
            temperature=0.1,
            timeout=60,
            max_tokens=max_tokens,
            max_retries=0,
        )
    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        return LLM(
            model=model,
            base_url=f"{host}/v1",
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            temperature=0.1,
            # Local models load into memory on first use; allow for that.
            timeout=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
            max_tokens=max_tokens,
            max_retries=0,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_llm_for(model: str) -> "LLM":
    """Build an LLM from a prefixed model string (gemini/, openrouter/, ollama/)."""

    provider = provider_of(model)
    if not provider:
        raise ValueError(f"Model string needs a provider prefix: {model}")
    return _build_llm(provider, model)


def crewai_llm() -> "LLM":
    """Create the primary configured CrewAI LLM client."""

    provider = os.environ["LLM_PROVIDER"].lower()
    return _build_llm(provider, os.environ["CREWAI_MODEL"])


def _chain_models() -> list[str]:
    return [m.strip() for m in os.getenv("LLM_FALLBACK_CHAIN", "").split(",") if m.strip()]


def fallback_crewai_llm() -> "LLM | None":
    """Return the first fallback provider (kept for callers that try one).

    The alternate provider is attempted only after the primary provider raises a
    transport/API exception. Invalid structured decisions still fail closed
    instead of being silently reinterpreted by another model.
    """

    chain = _chain_models()
    if chain:
        return build_llm_for(chain[0])
    primary = os.environ["LLM_PROVIDER"].lower()
    primary_model = os.environ["CREWAI_MODEL"]
    if primary == "openrouter" and os.getenv("GEMINI_API_KEY"):
        derived = primary_model.removeprefix("openrouter/google/")
        model = os.getenv("GEMINI_FALLBACK_MODEL", f"gemini/{derived}")
        return _build_llm("gemini", model)
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv(
        "OPEN_ROUTER_API_KEY"
    )
    if primary == "gemini" and openrouter_key:
        derived = primary_model.removeprefix("gemini/")
        model = os.getenv(
            "OPENROUTER_FALLBACK_MODEL",
            f"openrouter/google/{derived}",
        )
        return _build_llm("openrouter", model)
    return None


def extra_fallback_llms() -> list["LLM"]:
    """Fallbacks after the first one (only when LLM_FALLBACK_CHAIN lists several)."""

    built = []
    for model in _chain_models()[1:]:
        try:
            built.append(build_llm_for(model))
        except Exception:
            continue  # a misconfigured entry is skipped, never fatal
    return built


def narrative_llm() -> "LLM | None":
    """Optional cheaper model for narrative-only text (the run summary).

    ``LLM_NARRATIVE_MODEL=ollama/qwen2.5-coder:latest`` keeps the summary
    LLM-written but local and free; decisions keep using the primary chain.
    """

    model = os.getenv("LLM_NARRATIVE_MODEL", "").strip()
    return build_llm_for(model) if model else None
