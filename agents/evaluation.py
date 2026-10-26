"""CrewAI held-out evaluation agent and deterministic execution tool.

The evaluator accepts only the typed ``ModelSelectionResult`` plus MLflow
connection settings. It resolves the selected model and held-out dataset from
the selected run's artifacts, so raw rows never cross the agent boundary.
"""

from __future__ import annotations

import math
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import joblib
import pandas as pd
from crewai import Agent
from crewai.tools import BaseTool
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from contracts.models import (
    EvaluationResult,
    MetricDirection,
    ModelSelectionResult,
    TaskType,
)
from opentelemetry.trace import SpanKind

from orchestrator.llm_reasoning import generate_evaluation_reasoning
from orchestrator.telemetry import traced_span, update_span_io


DEFAULT_EVALUATION_ARTIFACT_DIR = Path("/tmp/artifacts/evaluation-cache")


def _default_tracking_uri() -> str:
    database = (Path("/tmp/artifacts") / "mlflow.db").resolve()
    return f"sqlite:///{database.as_posix()}"


class EvaluationError(ValueError):
    """Safe error raised when held-out evaluation cannot be completed."""


class _EvaluationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    task_type: TaskType
    target_column: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    test_artifact_file: str = Field(min_length=1)
    test_row_count: int = Field(gt=0)

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: list[str]) -> list[str]:
        if any(not column.strip() for column in value):
            raise ValueError("feature column names must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("feature column names must be unique")
        return value

    @field_validator("test_artifact_file")
    @classmethod
    def validate_test_artifact_file(cls, value: str) -> str:
        path = PurePath(value)
        if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
            raise ValueError("test_artifact_file must be a file name")
        return value


class EvaluationToolInput(BaseModel):
    """Validated input exposed by the CrewAI evaluation tool."""

    model_config = ConfigDict(extra="forbid")

    model_selection_result: ModelSelectionResult
    tracking_uri: str | None = Field(
        default=None,
        description="Optional MLflow URI; defaults to the pipeline's local store.",
    )
    acceptance_threshold: float | None = Field(
        default=None,
        description=(
            "Optional acceptance gate. Defaults to 0.80 weighted F1 for "
            "classification or 25% of held-out target standard deviation for RMSE."
        ),
    )

    @field_validator("tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"file", "http", "https", "sqlite"}:
            raise ValueError(
                "tracking_uri must use the file, sqlite, http, or https scheme"
            )
        if parsed.scheme in {"http", "https"} and not parsed.netloc:
            raise ValueError("HTTP tracking_uri must include a host")
        if parsed.scheme in {"file", "sqlite"} and not parsed.path:
            raise ValueError("local tracking_uri must include a path")
        return value

    @field_validator("acceptance_threshold")
    @classmethod
    def validate_threshold(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("acceptance_threshold must be finite")
        return value


def _download_evaluation_bundle(
    client: MlflowClient,
    selection: ModelSelectionResult,
    artifact_dir: Path,
) -> tuple[Any, pd.DataFrame, _EvaluationMetadata]:
    destination = artifact_dir.resolve() / selection.best_run_id
    destination.mkdir(parents=True, exist_ok=True)
    try:
        metadata_path = Path(
            client.download_artifacts(
                selection.best_run_id,
                "evaluation/metadata.json",
                str(destination),
            )
        )
        metadata = _EvaluationMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        test_path = Path(
            client.download_artifacts(
                selection.best_run_id,
                f"evaluation/{metadata.test_artifact_file}",
                str(destination),
            )
        )
        model_directory = Path(
            client.download_artifacts(
                selection.best_run_id,
                "model",
                str(destination),
            )
        )
    except (MlflowException, OSError, ValueError) as exc:
        raise EvaluationError(
            "selected MLflow run does not contain a valid evaluation bundle"
        ) from exc

    if metadata.run_id != selection.run_id:
        raise EvaluationError("evaluation metadata run_id does not match selection")
    model_files = sorted(model_directory.glob("*.joblib"))
    if len(model_files) != 1:
        raise EvaluationError("selected run must contain exactly one model artifact")
    try:
        model = joblib.load(model_files[0])
        held_out = pd.read_csv(test_path)
    except (OSError, ValueError, TypeError) as exc:
        raise EvaluationError("evaluation artifacts could not be loaded") from exc
    if not callable(getattr(model, "predict", None)):
        raise EvaluationError("selected model artifact does not support prediction")
    if len(held_out) != metadata.test_row_count:
        raise EvaluationError("held-out row count does not match evaluation metadata")
    required = set(metadata.feature_columns) | {metadata.target_column}
    if required - set(held_out.columns):
        raise EvaluationError("held-out artifact is missing contracted columns")
    if held_out[metadata.target_column].isna().any():
        raise EvaluationError("held-out target contains missing values")
    return model, held_out, metadata


def _classification_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    predictions: Any,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "f1_weighted": float(
            f1_score(target, predictions, average="weighted", zero_division=0)
        ),
    }
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba) and target.nunique() == 2:
        try:
            probabilities = predict_proba(features)
            auc = float(roc_auc_score(target, probabilities[:, 1]))
            if math.isfinite(auc):
                metrics["roc_auc"] = auc
        except (TypeError, ValueError, IndexError):
            pass
    return metrics


def _regression_metrics(target: pd.Series, predictions: Any) -> dict[str, float]:
    metrics = {
        "rmse": float(math.sqrt(mean_squared_error(target, predictions))),
        "mae": float(mean_absolute_error(target, predictions)),
    }
    if len(target) >= 2:
        r2 = float(r2_score(target, predictions))
        if math.isfinite(r2):
            metrics["r2"] = r2
    return metrics


def _evaluation_gate(
    task_type: TaskType,
    target: pd.Series,
    metrics: dict[str, float],
    requested_threshold: float | None,
) -> tuple[str, float, MetricDirection, str, bool, str | None]:
    if task_type == TaskType.CLASSIFICATION:
        threshold = 0.80 if requested_threshold is None else requested_threshold
        if not 0 <= threshold <= 1:
            raise EvaluationError(
                "classification acceptance_threshold must be between 0 and 1"
            )
        observed = metrics["f1_weighted"]
        passed = observed >= threshold
        rationale = (
            "Weighted F1 balances per-class precision and recall while accounting "
            f"for class support; the acceptance gate is {threshold:.4f}."
        )
        feedback = None
        if not passed:
            feedback = (
                f"Weighted F1 {observed:.4f} is below {threshold:.4f}. Retune "
                "class weights and decision thresholds, then inspect errors for "
                "under-represented classes before the next bounded attempt."
            )
        return (
            "f1_weighted",
            threshold,
            MetricDirection.MAXIMIZE,
            rationale,
            passed,
            feedback,
        )

    numeric_target = pd.to_numeric(target, errors="raise")
    if requested_threshold is None:
        target_scale = float(numeric_target.std(ddof=0))
        if not math.isfinite(target_scale):
            raise EvaluationError("regression target scale is not finite")
        threshold = max(target_scale * 0.25, 1e-12)
        rationale = (
            "RMSE penalizes large residuals; the acceptance gate is 25% of the "
            f"held-out target standard deviation ({threshold:.6g})."
        )
    else:
        threshold = requested_threshold
        rationale = (
            "RMSE penalizes large residuals; the caller supplied an acceptance "
            f"gate of {threshold:.6g}."
        )
    if threshold < 0:
        raise EvaluationError("regression acceptance_threshold cannot be negative")
    observed = metrics["rmse"]
    passed = observed <= threshold
    feedback = None
    if not passed:
        feedback = (
            f"RMSE {observed:.6g} exceeds {threshold:.6g}. Inspect the largest "
            "residuals and influential features, then retune regularization and "
            "tree depth before the next bounded attempt."
        )
    return (
        "rmse",
        threshold,
        MetricDirection.MINIMIZE,
        rationale,
        passed,
        feedback,
    )


class EvaluationReflection(BaseModel):
    """Structured output of the CrewAI evaluation agent (reflection step)."""

    model_config = ConfigDict(extra="forbid")

    diagnosis: str = Field(min_length=1, max_length=1_500)
    retry_feedback: str | None = Field(default=None, max_length=1_200)
    reasoning: str = Field(min_length=1, max_length=2_000)


def reflect_on_evaluation(
    *,
    task_type: TaskType,
    primary_metric: str,
    metrics: dict[str, float],
    threshold: float,
    passed: bool,
    gate_feedback: str | None,
    model_name: str,
) -> tuple[str, str | None, str]:
    """Reflection pattern: a CrewAI agent critiques the held-out result.

    The pass/fail gate is already decided deterministically. The agent explains
    the result and, on failure, writes concrete guidance that is sent back to
    the ADK model-selection agent for the next attempt.

    Returns (reasoning, retry_feedback, source).
    """

    lower_is_better = primary_metric in {"rmse", "mae", "mse"}
    observed = metrics.get(primary_metric)
    verdict = (
        f"PASSED: held-out {primary_metric} {observed:.6g} is "
        f"{'at or below the maximum' if lower_is_better else 'at or above the minimum'} "
        f"allowed {threshold:.6g}"
        if passed
        else f"FAILED: held-out {primary_metric} {observed:.6g} is "
        f"{'above the maximum' if lower_is_better else 'below the minimum'} allowed {threshold:.6g}"
    ) if observed is not None else ("PASSED" if passed else "FAILED")
    safe_input = {
        "task_type": str(task_type),
        "model_name": model_name,
        "primary_metric": primary_metric,
        "metric_direction": "lower is better" if lower_is_better else "higher is better",
        "held_out_metrics": metrics,
        "acceptance_threshold": threshold,
        "passed": passed,
        "verdict": verdict,
    }
    with traced_span(
        "llm.decision.evaluation_reflection",
        kind=SpanKind.CLIENT,
        attributes={"ml.reasoning.row_safe": True},
    ) as span:
        update_span_io(span, input=safe_input)
        try:
            import os

            if os.getenv("LLM_MODE", "live").strip().casefold() != "live":
                raise RuntimeError("LLM_MODE is not live")
            from orchestrator.crew_agents import run_crew_decision

            from orchestrator import decision_cache, usage

            def check(reflection: EvaluationReflection) -> None:
                if not passed and not (reflection.retry_feedback or "").strip():
                    raise ValueError("the model failed the gate: retry_feedback is required")

            reflection, source = None, "cache"
            cached = decision_cache.get("evaluation_reflection", safe_input)
            if cached is not None:
                try:
                    reflection = EvaluationReflection.model_validate(cached)
                    check(reflection)
                    usage.record(
                        stage="evaluation_reflection",
                        agent="Model Evaluation Engineer",
                        framework="crewai",
                        source="cache",
                    )
                except Exception:
                    reflection = None
            if reflection is None:
                reflection, source = run_crew_decision(
                    stage="evaluation_reflection",
                    role="Model Evaluation Engineer",
                    goal=(
                        "Judge whether the selected model generalises and, if it failed "
                        "the acceptance gate, tell the model-selection agent exactly what "
                        "to change on the next attempt."
                    ),
                    backstory=(
                        "You are a rigorous ML reviewer. You never change metric values "
                        "or the pass/fail decision; you interpret them and give "
                        "actionable, bounded advice (hyperparameters, regularisation, "
                        "class weighting, model family)."
                    ),
                    task_description=(
                        "The pass/fail verdict below is already decided and correct; do not "
                        "contradict it. Reflect on this held-out evaluation and return JSON with "
                        "diagnosis, retry_feedback (required if passed is false, else "
                        "null) and reasoning (3 short bullet points). Evidence: "
                        f"{safe_input}"
                    ),
                    expected_output="An EvaluationReflection JSON object.",
                    output_model=EvaluationReflection,
                    validator=check,
                )
                decision_cache.put("evaluation_reflection", safe_input, reflection.model_dump())
            feedback = None
            if not passed:
                feedback = f"{gate_feedback} Evaluator guidance: {reflection.retry_feedback}"
            reasoning = f"{reflection.diagnosis}\n{reflection.reasoning}"
            update_span_io(
                span,
                output={**reflection.model_dump(), "source": source, "fallback_used": False},
            )
            return reasoning, feedback, source
        except Exception as exc:
            from orchestrator import usage

            usage.record(
                stage="evaluation_reflection",
                agent="Model Evaluation Engineer",
                framework="crewai",
                source="fallback",
            )
            # Deterministic text: every provider already failed, so another
            # LLM call here would only repeat the failure.
            gate = "passed" if passed else "did not pass"
            next_step = "before deployment" if passed else "in the next retry"
            reasoning = (
                f"- The held-out metrics {gate} the configured acceptance gate of {threshold:.4f}.\n"
                "- The result reflects performance on data excluded from model fitting.\n"
                f"- Review residual or class-level errors {next_step}."
            )
            update_span_io(
                span,
                output={
                    "reasoning": reasoning,
                    "retry_feedback": gate_feedback,
                    "source": "fallback",
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
            return reasoning, gate_feedback, "fallback"


def evaluate_model(
    model_selection_result: ModelSelectionResult | dict[str, Any] | str,
    *,
    tracking_uri: str | None = None,
    acceptance_threshold: float | None = None,
    artifact_dir: str | Path = DEFAULT_EVALUATION_ARTIFACT_DIR,
) -> EvaluationResult:
    """Run a selected model once against its untouched held-out artifact."""

    try:
        if isinstance(model_selection_result, str):
            selection = ModelSelectionResult.model_validate_json(model_selection_result)
        else:
            selection = ModelSelectionResult.model_validate(model_selection_result)
    except ValueError as exc:
        raise EvaluationError("invalid ModelSelectionResult") from exc

    effective_tracking_uri = tracking_uri or _default_tracking_uri()
    try:
        input_payload = EvaluationToolInput(
            model_selection_result=selection,
            tracking_uri=effective_tracking_uri,
            acceptance_threshold=acceptance_threshold,
        )
    except ValueError as exc:
        raise EvaluationError("invalid evaluation settings") from exc

    client = MlflowClient(
        tracking_uri=input_payload.tracking_uri,
        registry_uri=input_payload.tracking_uri,
    )
    model, held_out, metadata = _download_evaluation_bundle(
        client,
        selection,
        Path(artifact_dir),
    )
    features = held_out[metadata.feature_columns]
    target = held_out[metadata.target_column]
    try:
        predictions = model.predict(features)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("selected model could not score the held-out artifact") from exc

    if metadata.task_type == TaskType.CLASSIFICATION:
        metrics = _classification_metrics(model, features, target, predictions)
    else:
        try:
            numeric_target = pd.to_numeric(target, errors="raise")
        except (TypeError, ValueError) as exc:
            raise EvaluationError("regression held-out target must be numeric") from exc
        metrics = _regression_metrics(numeric_target, predictions)

    (
        primary_metric,
        threshold,
        direction,
        rationale,
        passed,
        retry_feedback,
    ) = _evaluation_gate(
        metadata.task_type,
        target,
        metrics,
        input_payload.acceptance_threshold,
    )
    llm_reasoning, retry_feedback, _source = reflect_on_evaluation(
        task_type=metadata.task_type,
        primary_metric=primary_metric,
        metrics=metrics,
        threshold=threshold,
        passed=passed,
        gate_feedback=retry_feedback,
        model_name=selection.best_model_name,
    )
    return EvaluationResult(
        run_id=selection.run_id,
        model_uri=selection.best_model_uri,
        held_out_metrics=metrics,
        primary_metric=primary_metric,
        acceptance_threshold=threshold,
        metric_direction=direction,
        threshold_rationale=rationale,
        passed=passed,
        retry_feedback=retry_feedback,
        llm_reasoning=llm_reasoning,
    )


def run_evaluation(
    model_selection_result: ModelSelectionResult | dict[str, Any] | str,
    test_csv: str | Path,
    *,
    acceptance_threshold: float = 0.80,
) -> EvaluationResult:
    """Gate a mock or precomputed score for lightweight integration tests.

    Production evaluation should use :func:`evaluate_model`, which downloads
    and scores the selected model artifact. This helper intentionally supports
    smoke tests whose mock selection has no serialized MLflow model.
    """

    try:
        if isinstance(model_selection_result, str):
            selection = ModelSelectionResult.model_validate_json(model_selection_result)
        else:
            selection = ModelSelectionResult.model_validate(model_selection_result)
    except ValueError as exc:
        raise EvaluationError("invalid ModelSelectionResult") from exc

    csv_path = Path(test_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise EvaluationError("test CSV does not exist")
    try:
        held_out = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise EvaluationError("test CSV could not be read") from exc
    if held_out.empty:
        raise EvaluationError("test CSV must contain at least one row")
    if not 0 <= acceptance_threshold <= 1:
        raise EvaluationError("acceptance_threshold must be between 0 and 1")

    observed = selection.metric_value
    passed = observed >= acceptance_threshold
    feedback = None
    if not passed:
        feedback = (
            f"Selection score {observed:.4f} is below {acceptance_threshold:.4f}. "
            "Retune model hyperparameters and inspect held-out errors before retrying."
        )
    metrics = {"selection_score": observed}
    llm_reasoning = generate_evaluation_reasoning(
        metrics,
        passed,
        acceptance_threshold,
        feedback,
    )
    return EvaluationResult(
        run_id=selection.run_id,
        model_uri=selection.best_model_uri,
        held_out_metrics=metrics,
        primary_metric="selection_score",
        acceptance_threshold=acceptance_threshold,
        metric_direction=MetricDirection.MAXIMIZE,
        threshold_rationale=(
            "The integration gate requires the supplied selection score to meet "
            f"or exceed {acceptance_threshold:.4f}."
        ),
        passed=passed,
        retry_feedback=feedback,
        llm_reasoning=llm_reasoning,
    )


class EvaluateModelTool(BaseTool):
    """CrewAI tool that resolves MLflow lineage and runs held-out evaluation."""

    name: str = "evaluate_model"
    description: str = (
        "Load the selected MLflow model and untouched test artifact, calculate "
        "task-appropriate held-out metrics, and return EvaluationResult JSON."
    )
    args_schema: type[BaseModel] = EvaluationToolInput
    default_tracking_uri: str = Field(default_factory=_default_tracking_uri)
    artifact_dir: str = str(DEFAULT_EVALUATION_ARTIFACT_DIR)

    def _run(
        self,
        model_selection_result: ModelSelectionResult,
        tracking_uri: str | None = None,
        acceptance_threshold: float | None = None,
    ) -> str:
        result = evaluate_model(
            model_selection_result,
            tracking_uri=tracking_uri or self.default_tracking_uri,
            acceptance_threshold=acceptance_threshold,
            artifact_dir=self.artifact_dir,
        )
        return result.model_dump_json()


def build_evaluation_agent(
    *,
    llm: Any | None = None,
    tracking_uri: str | None = None,
    artifact_dir: str | Path = DEFAULT_EVALUATION_ARTIFACT_DIR,
) -> Agent:
    """Construct the CrewAI evaluator without making an LLM call."""

    tool = EvaluateModelTool(
        default_tracking_uri=tracking_uri or _default_tracking_uri(),
        artifact_dir=str(artifact_dir),
    )
    kwargs: dict[str, Any] = {
        "role": "Model Evaluation Engineer",
        "goal": (
            "Evaluate the selected model exactly once on untouched held-out data, "
            "apply a justified acceptance gate, and return actionable retry feedback."
        ),
        "backstory": (
            "You are a rigorous ML evaluator. You never inspect raw rows in the "
            "language model; you call evaluate_model and return its validated "
            "EvaluationResult without changing metric values or the decision."
        ),
        "tools": [tool],
        "allow_delegation": False,
        "verbose": False,
        "max_iter": 3,
    }
    if llm is not None:
        kwargs["llm"] = llm
    return Agent(**kwargs)
