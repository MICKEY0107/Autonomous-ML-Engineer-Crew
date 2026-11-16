"""Typed, row-safe evidence contracts for a completed pipeline run.

The dashboard and Cloud Storage use the same document.  This keeps the
assessment evidence independent of the ephemeral Cloud Run filesystem while
ensuring raw dataset rows and credentials never leave deterministic tools.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from contracts.models import ContractModel, MetricMap, NonBlankString


class EvidenceEvent(ContractModel):
    """One sanitized observable action during a pipeline run."""

    category: Literal["agent", "a2a", "mcp", "callback", "guardrail", "pipeline"]
    name: NonBlankString
    status: Literal["success", "failed", "running", "retry", "blocked", "passed"]
    duration_ms: int | None = Field(default=None, ge=0)
    source: str | None = None
    destination: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRun(ContractModel):
    """Portable evidence record rendered by the cloud evaluation dashboard."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: UUID
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: NonBlankString
    environment: dict[str, Any]
    dataset: dict[str, Any]
    model_quality: MetricMap = Field(default_factory=dict)
    primary_metric: str | None = None
    primary_metric_value: float | None = None
    primary_threshold: float | None = None
    pipeline: dict[str, Any]
    events: list[EvidenceEvent] = Field(default_factory=list)
    observability: dict[str, Any]
    cost: dict[str, Any]
    security: dict[str, Any]
    storage: dict[str, Any] = Field(default_factory=dict)


class EvaluationScenario(ContractModel):
    """One expected-versus-observed cloud evaluation scenario."""

    name: NonBlankString
    category: Literal["pipeline", "a2a", "mcp", "guardrail", "validation"]
    expected: NonBlankString
    observed: NonBlankString
    passed: bool
    duration_ms: int = Field(ge=0)
    run_id: UUID | None = None
    detail: str = ""


class EvaluationSuite(ContractModel):
    """Cloud Run Job result for the focused agentic-system test suite."""

    schema_version: Literal["1.0"] = "1.0"
    suite_id: NonBlankString
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_url: NonBlankString
    environment: dict[str, Any]
    scenarios: list[EvaluationScenario] = Field(min_length=1)

    @property
    def passed_count(self) -> int:
        return sum(1 for scenario in self.scenarios if scenario.passed)

