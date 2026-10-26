"""Pydantic contracts for every pipeline state and agent hand-off.

The contracts deliberately contain artifact references and summaries only. Raw
dataset rows must remain inside deterministic data tools and never cross an
agent, MCP, or A2A boundary.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ArtifactUri = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
JsonScalar = str | int | float | bool | None
MetricMap = dict[str, float]

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")
_TRACE_ID_PATTERN = re.compile(r"^[a-fA-F0-9]{16,32}$")


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class PipelineStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUCCESS = "SUCCESS"
    FAILED_EVAL_EXHAUSTED = "FAILED_EVAL_EXHAUSTED"
    FAILED_INPUT = "FAILED_INPUT"


class PipelineStage(StrEnum):
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    MODEL_SELECTION = "model_selection"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    COMPLETE = "complete"


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ContractModel(BaseModel):
    """Common fail-closed settings for external boundary models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )


def _validate_artifact_uri(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme:
        raise ValueError("must be an absolute URI with a scheme")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ValueError("HTTP artifact URI must include a host")
    if parsed.scheme == "file" and not parsed.path:
        raise ValueError("file artifact URI must include a path")
    return value


def _validate_metric_map(value: MetricMap) -> MetricMap:
    if not value:
        raise ValueError("must contain at least one metric")
    if any(not name.strip() for name in value):
        raise ValueError("metric names must not be blank")
    if any(not math.isfinite(metric) for metric in value.values()):
        raise ValueError("metric values must be finite")
    return value


class DataPrepResult(ContractModel):
    """Validated output of the CrewAI data-preparation agent."""

    run_id: UUID
    dataset_hash: str = Field(description="Lowercase SHA-256 of the source dataset.")
    schema_summary: dict[NonBlankString, NonBlankString] = Field(min_length=1)
    quality_findings: list[NonBlankString] = Field(default_factory=list)
    target_column: NonBlankString
    feature_columns: list[NonBlankString] = Field(min_length=1)
    row_count: int = Field(gt=0)
    train_row_count: int = Field(gt=0)
    test_row_count: int = Field(gt=0)
    train_artifact_uri: ArtifactUri
    test_artifact_uri: ArtifactUri
    llm_reasoning: str = ""

    @property
    def train_path(self) -> str:
        """Return the local path represented by ``train_artifact_uri``."""

        parsed = urlparse(self.train_artifact_uri)
        if parsed.scheme != "file":
            raise ValueError("train artifact is not a local file URI")
        path = parsed.path
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        return url2pathname(unquote(path))

    @property
    def test_path(self) -> str:
        """Return the local path represented by ``test_artifact_uri``."""

        parsed = urlparse(self.test_artifact_uri)
        if parsed.scheme != "file":
            raise ValueError("test artifact is not a local file URI")
        path = parsed.path
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        return url2pathname(unquote(path))

    @field_validator("dataset_hash")
    @classmethod
    def validate_dataset_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("dataset_hash must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("train_artifact_uri", "test_artifact_uri")
    @classmethod
    def validate_artifact_uris(cls, value: str) -> str:
        return _validate_artifact_uri(value)

    @model_validator(mode="after")
    def validate_dataset_shape(self) -> Self:
        if self.target_column not in self.schema_summary:
            raise ValueError("target_column must be present in schema_summary")
        if self.target_column in self.feature_columns:
            raise ValueError("target_column must not be included in feature_columns")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique")
        unknown_features = set(self.feature_columns) - set(self.schema_summary)
        if unknown_features:
            raise ValueError("every feature column must be present in schema_summary")
        if self.train_row_count + self.test_row_count > self.row_count:
            raise ValueError("train and test rows cannot exceed the source row count")
        if self.train_artifact_uri == self.test_artifact_uri:
            raise ValueError("train and test artifacts must be different")
        return self


class ModelSelectionResult(ContractModel):
    """Validated A2A response from the Google ADK model-selection agent."""

    run_id: UUID
    candidate_run_ids: list[NonBlankString] = Field(min_length=1)
    best_run_id: NonBlankString
    best_model_uri: ArtifactUri
    best_parameters: dict[NonBlankString, JsonScalar] = Field(default_factory=dict)
    training_metrics: MetricMap
    llm_reasoning: str = ""
    # Token/latency records of the ADK agent's LLM decisions (no row data).
    llm_usage: list[dict[str, Any]] = Field(default_factory=list)
    # Every trained candidate of this attempt: params, metrics, quality band, pick.
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    retry_attempt: int = Field(default=0, ge=0)

    @property
    def best_model_name(self) -> str:
        """Human-readable selected model name derived from tracked parameters."""

        model_type = self.best_parameters.get("model_type")
        if isinstance(model_type, str) and model_type.strip():
            return model_type
        return self.best_model_uri.rstrip("/").rsplit("/", maxsplit=1)[-1]

    @property
    def metric_value(self) -> float:
        """Primary selection score using a stable metric preference order."""

        for metric_name in ("f1_weighted", "accuracy", "roc_auc", "r2", "rmse"):
            if metric_name in self.training_metrics:
                return float(self.training_metrics[metric_name])
        return float(next(iter(self.training_metrics.values())))

    @field_validator("best_model_uri")
    @classmethod
    def validate_model_uri(cls, value: str) -> str:
        return _validate_artifact_uri(value)

    @field_validator("training_metrics")
    @classmethod
    def validate_training_metrics(cls, value: MetricMap) -> MetricMap:
        return _validate_metric_map(value)

    @model_validator(mode="after")
    def validate_best_candidate(self) -> Self:
        if len(set(self.candidate_run_ids)) != len(self.candidate_run_ids):
            raise ValueError("candidate_run_ids must be unique")
        if self.best_run_id not in self.candidate_run_ids:
            raise ValueError("best_run_id must identify one of candidate_run_ids")
        return self


class EvaluationResult(ContractModel):
    """Held-out evaluation and bounded-retuning decision."""

    run_id: UUID
    model_uri: ArtifactUri
    held_out_metrics: MetricMap
    primary_metric: NonBlankString
    acceptance_threshold: float
    metric_direction: MetricDirection
    threshold_rationale: NonBlankString
    passed: bool
    retry_feedback: NonBlankString | None = None
    llm_reasoning: str = ""

    @property
    def feedback(self) -> str:
        """Return retry guidance, or the passing gate rationale when accepted."""

        return self.retry_feedback or self.threshold_rationale

    @field_validator("model_uri")
    @classmethod
    def validate_model_uri(cls, value: str) -> str:
        return _validate_artifact_uri(value)

    @field_validator("held_out_metrics")
    @classmethod
    def validate_held_out_metrics(cls, value: MetricMap) -> MetricMap:
        return _validate_metric_map(value)

    @field_validator("acceptance_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("acceptance_threshold must be finite")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.primary_metric not in self.held_out_metrics:
            raise ValueError("primary_metric must be present in held_out_metrics")
        observed = self.held_out_metrics[self.primary_metric]
        expected_pass = (
            observed >= self.acceptance_threshold
            if self.metric_direction == MetricDirection.MAXIMIZE
            else observed <= self.acceptance_threshold
        )
        if self.passed != expected_pass:
            raise ValueError("passed is inconsistent with the metric and threshold")
        if not self.passed and self.retry_feedback is None:
            raise ValueError("retry_feedback is required when evaluation fails")
        return self


class DeploymentResult(ContractModel):
    """Deployment identity and smoke-test result."""

    run_id: UUID
    image_digest: str
    service_url: HttpUrl
    health_status: HealthStatus
    deployment_revision: NonBlankString
    deployed_at: datetime

    @field_validator("image_digest")
    @classmethod
    def validate_image_digest(cls, value: str) -> str:
        if not _IMAGE_DIGEST_PATTERN.fullmatch(value):
            raise ValueError("image_digest must use the sha256:<64 lowercase hex> form")
        return value

    @field_validator("deployed_at")
    @classmethod
    def validate_deployed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("deployed_at must be timezone-aware")
        return value


class PipelineState(ContractModel):
    """Single source of mutable context owned by the CrewAI Flow."""

    run_id: UUID
    dataset_uri: ArtifactUri
    task_type: TaskType
    target_column: NonBlankString
    prediction_goal: NonBlankString
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    artifact_references: dict[NonBlankString, ArtifactUri] = Field(default_factory=dict)
    trace_id: str
    data_prep_result: DataPrepResult | None = None
    model_selection_result: ModelSelectionResult | None = None
    evaluation_result: EvaluationResult | None = None
    deployment_result: DeploymentResult | None = None
    errors: list[NonBlankString] = Field(default_factory=list)

    @field_validator("dataset_uri")
    @classmethod
    def validate_dataset_uri(cls, value: str) -> str:
        return _validate_artifact_uri(value)

    @field_validator("artifact_references")
    @classmethod
    def validate_artifact_references(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        return {name: _validate_artifact_uri(uri) for name, uri in value.items()}

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        if not _TRACE_ID_PATTERN.fullmatch(value):
            raise ValueError("trace_id must contain 16 to 32 hexadecimal characters")
        return value.lower()

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.retry_count > self.max_retries:
            raise ValueError("retry_count cannot exceed max_retries")

        results = (
            self.data_prep_result,
            self.model_selection_result,
            self.evaluation_result,
            self.deployment_result,
        )
        if any(result is not None and result.run_id != self.run_id for result in results):
            raise ValueError("all hand-off results must use the pipeline run_id")

        if self.status == PipelineStatus.COMPLETED:
            if self.current_stage != PipelineStage.COMPLETE:
                raise ValueError("a completed pipeline must be in the complete stage")
            if any(result is None for result in results):
                raise ValueError("a completed pipeline requires every stage result")
            if not self.evaluation_result or not self.evaluation_result.passed:
                raise ValueError("a completed pipeline requires a passing evaluation")
            if (
                not self.deployment_result
                or self.deployment_result.health_status != HealthStatus.HEALTHY
            ):
                raise ValueError("a completed pipeline requires a healthy deployment")
        if self.status == PipelineStatus.FAILED and not self.errors:
            raise ValueError("a failed pipeline must include at least one sanitized error")
        if self.status == PipelineStatus.SUCCESS:
            if self.current_stage != PipelineStage.COMPLETE:
                raise ValueError("a successful orchestration must be in the complete stage")
            if (
                self.data_prep_result is None
                or self.model_selection_result is None
                or self.evaluation_result is None
                or not self.evaluation_result.passed
            ):
                raise ValueError(
                    "a successful orchestration requires passing results through evaluation"
                )
        if self.status in {
            PipelineStatus.FAILED_EVAL_EXHAUSTED,
            PipelineStatus.FAILED_INPUT,
        } and not self.errors:
            raise ValueError("a terminally failed orchestration must include an error")
        return self


class PipelineResult(ContractModel):
    """Terminal machine-readable result used to render the model report card."""

    run_id: UUID
    status: Literal["completed", "evaluation_completed", "failed"]
    model_card_markdown: NonBlankString
    endpoint: HttpUrl | None = None
    lineage: dict[NonBlankString, NonBlankString] = Field(min_length=1)
    trace_url: HttpUrl
    evaluation: EvaluationResult | None = None
    deployment: DeploymentResult | None = None
    runtime_seconds: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    # Per-agent tokens, requests, latency and cost (see orchestrator/usage.py).
    usage_report: dict[str, Any] = Field(default_factory=dict)
    error: NonBlankString | None = None
    llm_summary: str = ""
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("runtime_seconds", "estimated_cost_usd")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_terminal_result(self) -> Self:
        if self.evaluation is not None and self.evaluation.run_id != self.run_id:
            raise ValueError("evaluation must use the pipeline run_id")
        if self.deployment is not None and self.deployment.run_id != self.run_id:
            raise ValueError("deployment must use the pipeline run_id")

        if self.status == "completed":
            if self.error is not None:
                raise ValueError("a completed result cannot contain an error")
            if self.endpoint is None or self.evaluation is None or self.deployment is None:
                raise ValueError(
                    "a completed result requires endpoint, evaluation, and deployment"
                )
            if not self.evaluation.passed:
                raise ValueError("a completed result requires a passing evaluation")
            if self.deployment.health_status != HealthStatus.HEALTHY:
                raise ValueError("a completed result requires a healthy deployment")
            if str(self.endpoint).rstrip("/") != str(self.deployment.service_url).rstrip("/"):
                raise ValueError("endpoint must match the deployed service URL")
        elif self.status == "evaluation_completed":
            if self.error is not None:
                raise ValueError("an evaluation-completed result cannot contain an error")
            if self.evaluation is None or not self.evaluation.passed:
                raise ValueError(
                    "an evaluation-completed result requires a passing evaluation"
                )
            if self.endpoint is not None or self.deployment is not None:
                raise ValueError(
                    "an evaluation-completed result cannot claim a deployment"
                )
        elif self.error is None:
            raise ValueError("a failed result must include a sanitized error")
        return self
