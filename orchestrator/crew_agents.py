"""CrewAI agents that make the orchestrator-side LLM decisions.

Each decision is a real CrewAI execution: ``Agent`` + ``Task`` + ``Crew``
(sequential process) with

* ``output_pydantic`` - the LLM's answer must parse into a typed contract
  (structured output), and
* ``guardrail``       - a CrewAI task guardrail that validates the answer
  against the run's real metadata (e.g. no unknown / target columns) and
  makes the agent retry if it is invalid.

The agents only ever see row-free metadata (schema, counts, metrics). If the
primary provider fails, the secondary provider is tried; if both fail, the
caller falls back to a typed deterministic policy, so the pipeline never
stops because an LLM API is down.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from orchestrator import usage

# Never pause a run on CrewAI's interactive "share traces?" prompt; our
# observability is Langfuse/OpenTelemetry.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

T = TypeVar("T", bound=BaseModel)

GuardrailFn = Callable[[Any], tuple[bool, Any]]


def _provider_chain() -> list[tuple[str, Any]]:
    """Primary, then every configured fallback (cloud or local Ollama)."""

    from orchestrator.llm_provider import (
        crewai_llm,
        extra_fallback_llms,
        fallback_crewai_llm,
    )

    chain: list[tuple[str, Any]] = []
    for source, factory in (("llm", crewai_llm), ("secondary_llm", fallback_crewai_llm)):
        try:
            built = factory()
        except Exception:
            built = None  # not constructible (e.g. missing key): skip it
        if built is not None:
            chain.append((source, built))
    chain.extend(("secondary_llm", built) for built in extra_fallback_llms())
    return chain


def pydantic_guardrail(
    output_model: type[T],
    validator: Callable[[T], None] | None = None,
) -> GuardrailFn:
    """Build a CrewAI task guardrail: (passed, value_or_error_message)."""

    def guardrail(task_output):  # CrewAI inspects this signature; keep it unannotated
        try:
            parsed = task_output.pydantic
            if parsed is None:
                parsed = output_model.model_validate_json(task_output.raw)
            parsed = output_model.model_validate(parsed.model_dump())
            if validator is not None:
                validator(parsed)
        except Exception as exc:  # message is sent back to the agent to fix
            return False, f"Invalid decision, fix and answer again: {exc}"
        return True, task_output

    return guardrail


def run_crew_decision(
    *,
    stage: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    expected_output: str,
    output_model: type[T],
    validator: Callable[[T], None] | None = None,
) -> tuple[T, str]:
    """Kick off a one-agent CrewAI crew and return (typed decision, source).

    Raises the last provider error if every configured provider fails.
    """

    from crewai import Agent, Crew, Process, Task

    last_error: Exception | None = None
    for source, llm in _provider_chain():
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            allow_delegation=False,
            verbose=False,
            max_iter=2,
            # Fail fast on provider errors: move to the next provider instead
            # of re-running the same overloaded model.
            max_retry_limit=0,
        )
        task = Task(
            description=task_description,
            expected_output=expected_output,
            agent=agent,
            output_pydantic=output_model,
            guardrail=pydantic_guardrail(output_model, validator),
            guardrail_max_retries=1,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        started = time.perf_counter()
        try:
            output = crew.kickoff()
        except Exception as exc:
            last_error = exc
            continue
        elapsed = time.perf_counter() - started
        metrics = output.token_usage
        usage.record(
            stage=stage,
            agent=role,
            framework="crewai",
            provider=str(getattr(llm, "provider", "") or os.getenv("LLM_PROVIDER", "")),
            model=str(getattr(llm, "model", "")),
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens,
            requests=metrics.successful_requests,
            latency_seconds=round(elapsed, 3),
            source=source,
        )
        parsed = output.pydantic
        if parsed is None:
            parsed = output_model.model_validate_json(output.raw)
        decision = output_model.model_validate(parsed.model_dump())
        if validator is not None:
            validator(decision)
        return decision, source
    raise last_error or RuntimeError("no LLM provider configured")
