"""Configured CrewAI LLM providers for agentic pipeline decisions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from crewai import LLM


load_dotenv()


def _build_llm(provider: str, model: str) -> "LLM":
    from crewai import LLM

    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "700"))
    if provider == "gemini":
        from google.genai import types

        timeout_ms = int(os.getenv("LLM_FALLBACK_TIMEOUT_SECONDS", "15")) * 1_000
        return LLM(
            model=model,
            api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.1,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            client_params={"http_options": types.HttpOptions(timeout=timeout_ms)},
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
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def crewai_llm() -> "LLM":
    """Create the primary configured CrewAI LLM client."""

    provider = os.environ["LLM_PROVIDER"].lower()
    return _build_llm(provider, os.environ["CREWAI_MODEL"])


def fallback_crewai_llm() -> "LLM | None":
    """Return an independently configured secondary provider when available.

    The alternate provider is attempted only after the primary provider raises a
    transport/API exception. Invalid structured decisions still fail closed
    instead of being silently reinterpreted by another model.
    """

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
