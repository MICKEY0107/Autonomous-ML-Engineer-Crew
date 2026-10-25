"""Per-run LLM usage ledger: tokens, requests, latency and estimated cost.

Every live LLM decision (CrewAI crews on the orchestrator side, and the ADK
agent's planner calls inside the A2A service) appends one ``LLMUsageRecord``.
The ADK service returns its records inside ``ModelSelectionResult.llm_usage``
so the orchestrator can total the whole pipeline in ``PipelineResult``.

Cost is an *estimate*: tokens x the per-million prices configured with
``LLM_INPUT_COST_PER_1M`` / ``LLM_OUTPUT_COST_PER_1M`` (USD). Nothing is
hardcoded; with no prices configured the cost is reported as 0.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMUsageRecord(BaseModel):
    """One live LLM call (or one CrewAI crew kickoff)."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    agent: str
    framework: str = Field(description="crewai | adk")
    provider: str = ""
    model: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    source: str = Field(default="llm", description="llm | secondary_llm")


_LOCK = threading.Lock()
_RECORDS: list[LLMUsageRecord] = []


def _price(name: str) -> float:
    try:
        return max(float(os.getenv(name, "0") or 0), 0.0)
    except ValueError:
        return 0.0


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens / 1_000_000 * _price("LLM_INPUT_COST_PER_1M")
        + completion_tokens / 1_000_000 * _price("LLM_OUTPUT_COST_PER_1M"),
        8,
    )


def reset() -> None:
    """Start a fresh ledger (one pipeline run at a time per process)."""

    with _LOCK:
        _RECORDS.clear()


def record(**fields: Any) -> LLMUsageRecord:
    prompt = int(fields.get("prompt_tokens", 0) or 0)
    completion = int(fields.get("completion_tokens", 0) or 0)
    fields.setdefault("total_tokens", prompt + completion)
    fields.setdefault("estimated_cost_usd", estimate_cost(prompt, completion))
    entry = LLMUsageRecord.model_validate(fields)
    with _LOCK:
        _RECORDS.append(entry)
    return entry


def extend(entries: list[dict[str, Any]] | list[LLMUsageRecord]) -> None:
    validated = [LLMUsageRecord.model_validate(entry) for entry in entries]
    with _LOCK:
        _RECORDS.extend(validated)


def records() -> list[LLMUsageRecord]:
    with _LOCK:
        return list(_RECORDS)


def drain() -> list[LLMUsageRecord]:
    """Return and clear the records (used by the ADK service per request)."""

    with _LOCK:
        taken = list(_RECORDS)
        _RECORDS.clear()
        return taken


def summary(entries: list[LLMUsageRecord] | None = None) -> dict[str, Any]:
    entries = records() if entries is None else entries
    by_agent: dict[str, dict[str, float]] = {}
    for entry in entries:
        bucket = by_agent.setdefault(
            entry.agent,
            {"total_tokens": 0, "requests": 0, "estimated_cost_usd": 0.0, "latency_seconds": 0.0},
        )
        bucket["total_tokens"] += entry.total_tokens
        bucket["requests"] += entry.requests
        bucket["estimated_cost_usd"] = round(bucket["estimated_cost_usd"] + entry.estimated_cost_usd, 8)
        bucket["latency_seconds"] = round(bucket["latency_seconds"] + entry.latency_seconds, 3)
    return {
        "input_tokens": sum(e.prompt_tokens for e in entries),
        "output_tokens": sum(e.completion_tokens for e in entries),
        "total_tokens": sum(e.total_tokens for e in entries),
        "llm_requests": sum(e.requests for e in entries),
        "llm_latency_seconds": round(sum(e.latency_seconds for e in entries), 3),
        "estimated_cost_usd": round(sum(e.estimated_cost_usd for e in entries), 8),
        "pricing": {
            "input_per_1m_usd": _price("LLM_INPUT_COST_PER_1M"),
            "output_per_1m_usd": _price("LLM_OUTPUT_COST_PER_1M"),
        },
        "by_agent": by_agent,
        "calls": [e.model_dump() for e in entries],
    }
