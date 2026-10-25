"""Agentic Google ADK model-selection agent exposed over A2A.

Run the module to start the A2A JSON-RPC service::

    python -m agents.model_selection_adk

The incoming A2A message must contain JSON matching
``ModelSelectionRequest``. The response text is validated
``ModelSelectionResult`` JSON.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
import os
import re
import time
from collections.abc import AsyncGenerator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import joblib
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mlflow import MlflowClient
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from starlette.requests import Request
from starlette.responses import JSONResponse

from contracts.models import DataPrepResult, ModelSelectionResult, TaskType
from tools.log_experiment import (
    ExperimentResult,
    server as experiment_mcp_server,
)
from orchestrator.telemetry import (
    extracted_trace_context,
    set_current_span_attributes,
    traced_span,
    update_span_io,
)
from orchestrator.agentic_policy import ModelSearchPlan, choose_model, plan_model_search


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

A2A_HOST = os.environ["A2A_BIND_HOST"]
A2A_PORT = int(os.environ["A2A_BIND_PORT"])
DEFAULT_MODEL_ARTIFACT_DIR = Path("/tmp/artifacts")
_T = TypeVar("_T")


class ModelSelectionError(ValueError):
    """Safe error raised when a model-selection task cannot be completed."""


def _default_tracking_uri() -> str:
    database = (DEFAULT_MODEL_ARTIFACT_DIR / "mlflow.db").resolve()
    return f"sqlite:///{database.as_posix()}"


class ModelSelectionRequest(BaseModel):
    """Typed JSON payload accepted in an A2A task message."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    data_prep_result: DataPrepResult
    task_type: TaskType
    tracking_uri: str = Field(default_factory=_default_tracking_uri, min_length=1)
    experiment_name: str = Field(default="autonomous-model-selection", min_length=1)
    registered_model_name: str | None = Field(default=None, min_length=1)
    validation_size: float = Field(default=0.25, gt=0, lt=1)
    random_seed: int = Field(default=42, ge=0)
    retry_feedback: str | None = Field(default=None, min_length=1)
    retry_attempt: int = Field(default=0, ge=0, le=2)
    prediction_goal: str = Field(
        default="maximize validated predictive quality",
        min_length=1,
        max_length=2_000,
    )

    @field_validator("tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, value: str) -> str:
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


@dataclass(slots=True)
class _CandidateOutcome:
    name: str
    pipeline: Pipeline
    parameters: dict[str, str | int | float | bool]
    metrics: dict[str, float]
    training_time_ms: int


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ModelSelectionError("training artifact must use a local file URI")

    path_text = unquote(parsed.path)
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    elif re.match(r"^/[A-Za-z]:/", path_text):
        path_text = path_text[1:]
    path = Path(url2pathname(path_text)).resolve()
    if not path.is_file():
        raise ModelSelectionError("training artifact does not exist")
    return path


def _load_training_frame(data_prep: DataPrepResult) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(_path_from_file_uri(data_prep.train_artifact_uri))
    required_columns = [*data_prep.feature_columns, data_prep.target_column]
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ModelSelectionError("training artifact is missing contracted columns")
    if len(frame) != data_prep.train_row_count:
        raise ModelSelectionError("training artifact row count does not match its contract")
    if len(frame) < 4:
        raise ModelSelectionError("at least four training rows are required")

    features = frame[data_prep.feature_columns].copy()
    target = frame[data_prep.target_column].copy()
    if target.isna().any():
        raise ModelSelectionError("training target contains missing values")
    return features, target


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ModelSelectionError("training artifact has no feature columns")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _candidate_estimators(
    task_type: TaskType,
    random_seed: int,
    retry_attempt: int,
    search_plan: ModelSearchPlan,
) -> list[tuple[str, Any, dict[str, str | int | float | bool]]]:
    configured: list[tuple[str, Any, dict[str, str | int | float | bool]]] = []
    for candidate in search_plan.candidates:
        params = dict(candidate.parameters)
        name = candidate.name
        if name == "logistic_regression":
            estimator = LogisticRegression(
                C=float(params["C"]),
                max_iter=int(params["max_iter"]),
                random_state=random_seed,
            )
        elif name == "random_forest" and task_type == TaskType.CLASSIFICATION:
            estimator = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=(
                    None if params["max_depth"] is None else int(params["max_depth"])
                ),
                class_weight=params["class_weight"],
                random_state=random_seed,
                n_jobs=1,
            )
        elif name == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                random_state=random_seed,
            )
        elif name == "linear_regression":
            estimator = LinearRegression()
        elif name == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=(
                    None if params["max_depth"] is None else int(params["max_depth"])
                ),
                random_state=random_seed,
                n_jobs=1,
            )
        elif name == "ridge":
            estimator = Ridge(alpha=float(params["alpha"]))
        else:
            raise ModelSelectionError(f"unsupported planned candidate: {name}")
        logged_params: dict[str, str | int | float | bool] = {
            key: value
            for key, value in params.items()
            if value is not None
        }
        logged_params.update({
            "random_state": random_seed,
            "retry_attempt": retry_attempt,
        })
        configured.append((name, estimator, logged_params))
    return configured


def _classification_stratify(target: pd.Series, validation_size: float) -> pd.Series | None:
    counts = target.value_counts(dropna=False)
    class_count = len(counts)
    validation_rows = math.ceil(len(target) * validation_size)
    training_rows = len(target) - validation_rows
    if (
        class_count >= 2
        and counts.min() >= 2
        and validation_rows >= class_count
        and training_rows >= class_count
    ):
        return target
    return None


def _split_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    request: ModelSelectionRequest,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if request.task_type == TaskType.CLASSIFICATION and target.nunique() < 2:
        raise ModelSelectionError("classification training target needs at least two classes")
    if request.task_type == TaskType.REGRESSION:
        try:
            target = pd.to_numeric(target, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ModelSelectionError("regression target must be numeric") from exc

    stratify = (
        _classification_stratify(target, request.validation_size)
        if request.task_type == TaskType.CLASSIFICATION
        else None
    )
    try:
        return train_test_split(
            features,
            target,
            test_size=request.validation_size,
            random_state=request.random_seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ModelSelectionError(
            "training artifact cannot produce the requested validation split"
        ) from exc


def _evaluate_candidate(
    task_type: TaskType,
    expected: pd.Series,
    predicted: Any,
) -> dict[str, float]:
    if task_type == TaskType.CLASSIFICATION:
        return {
            "accuracy": float(accuracy_score(expected, predicted)),
            "f1_weighted": float(
                f1_score(expected, predicted, average="weighted", zero_division=0)
            ),
        }

    metrics = {
        "rmse": float(math.sqrt(mean_squared_error(expected, predicted))),
        "mae": float(mean_absolute_error(expected, predicted)),
    }
    if len(expected) >= 2:
        r2 = float(r2_score(expected, predicted))
        if math.isfinite(r2):
            metrics["r2"] = r2
    return metrics


def _train_candidates(request: ModelSelectionRequest) -> tuple[
    list[_CandidateOutcome],
    pd.DataFrame,
    pd.Series,
    str,
]:
    features, target = _load_training_frame(request.data_prep_result)
    search_plan = plan_model_search(
        task_type=request.task_type.value,
        row_count=len(features),
        feature_count=len(features.columns),
        prediction_goal=request.prediction_goal,
        retry_attempt=request.retry_attempt,
        retry_feedback=request.retry_feedback,
    )
    train_features, validation_features, train_target, validation_target = (
        _split_training_data(features, target, request)
    )
    outcomes: list[_CandidateOutcome] = []
    for name, estimator, parameters in _candidate_estimators(
        request.task_type,
        request.random_seed,
        request.retry_attempt,
        search_plan,
    ):
        pipeline = Pipeline(
            [
                ("preprocessor", _preprocessor(train_features)),
                ("estimator", estimator),
            ]
        )
        training_started = time.perf_counter()
        try:
            pipeline.fit(train_features, train_target)
            predictions = pipeline.predict(validation_features)
        except (TypeError, ValueError) as exc:
            raise ModelSelectionError(f"{name} could not be trained") from exc
        metrics = _evaluate_candidate(
            request.task_type,
            validation_target,
            predictions,
        )
        outcomes.append(
            _CandidateOutcome(
                name=name,
                pipeline=pipeline,
                parameters={"model_type": name, **parameters},
                metrics=metrics,
                training_time_ms=max(
                    1,
                    int((time.perf_counter() - training_started) * 1_000),
                ),
            )
        )
    return (
        outcomes,
        features,
        target,
        f"{search_plan.source.upper()} policy: {search_plan.reasoning}",
    )


def _structured_result(response: Any) -> ExperimentResult:
    structured = getattr(
        response,
        "structuredContent",
        getattr(response, "structured_content", None),
    )
    is_error = getattr(response, "isError", getattr(response, "is_error", False))
    if is_error or not isinstance(structured, dict):
        raise ModelSelectionError(
            "log_and_query_experiment MCP tool did not return a valid result"
        )
    try:
        # MCP structured content is JSON-shaped, so enum values arrive as
        # strings. Pydantic still validates every field and forbids extras;
        # Python strict mode would incorrectly reject valid JSON enum values.
        return ExperimentResult.model_validate(structured)
    except ValidationError as exc:
        raise ModelSelectionError(
            "log_and_query_experiment MCP result failed contract validation"
        ) from exc


async def _experiment_call(
    session: ClientSession,
    tracking_uri: str,
    action: str,
    **arguments: Any,
) -> ExperimentResult:
    with traced_span(
        "mcp.log_and_query_experiment",
        kind=SpanKind.CLIENT,
        attributes={
            "rpc.system": "mcp",
            "rpc.method": "log_and_query_experiment",
            "mcp.transport": "memory",
            "mcp.action": action,
            "mcp.arguments": arguments,
        },
    ) as span:
        update_span_io(
            span,
            input={
                "experiment_name": arguments.get("experiment_name"),
                "params": arguments.get("params", {}),
            },
        )
        response = await session.call_tool(
            "log_and_query_experiment",
            {"action": action, "tracking_uri": tracking_uri, **arguments},
        )
        result = _structured_result(response)
        set_current_span_attributes(
            {
                "mcp.result.validated": True,
                "mlflow.run_id": result.run_id,
                "mlflow.experiment_id": result.experiment_id,
            }
        )
        update_span_io(
            span,
            output={
                "run_id": result.run_id,
                "best_metric": next(iter(result.metrics.values()), None),
            },
        )
        return result


async def log_and_query_experiment(
    request: ModelSelectionRequest,
    outcomes: list[_CandidateOutcome],
    features: pd.DataFrame,
    target: pd.Series,
    artifact_dir: Path,
    search_reasoning: str,
) -> ModelSelectionResult:
    """Log all candidates through MCP, query the winner, and register it."""

    experiment_name = f"{request.experiment_name}-{request.data_prep_result.run_id.hex}"
    primary_metric = (
        "f1_weighted"
        if request.task_type == TaskType.CLASSIFICATION
        else "rmse"
    )
    maximize = request.task_type == TaskType.CLASSIFICATION
    candidate_run_ids: list[str] = []
    outcome_by_run_id: dict[str, _CandidateOutcome] = {}
    run_id_by_name: dict[str, str] = {}
    created_by_run_id: dict[str, ExperimentResult] = {}

    async with create_connected_server_and_client_session(
        experiment_mcp_server,
        raise_exceptions=True,
    ) as session:
        for outcome in outcomes:
            created = await _experiment_call(
                session,
                request.tracking_uri,
                "create_run",
                experiment_name=experiment_name,
                run_name=outcome.name,
                tags={
                    "pipeline_run_id": str(request.data_prep_result.run_id),
                    "task_type": request.task_type.value,
                    "retry_attempt": str(request.retry_attempt),
                    "retry_feedback": request.retry_feedback or "initial selection",
                },
            )
            candidate_run_ids.append(created.run_id)
            outcome_by_run_id[created.run_id] = outcome
            run_id_by_name[outcome.name] = created.run_id
            created_by_run_id[created.run_id] = created
            await _experiment_call(
                session,
                request.tracking_uri,
                "log_params",
                run_id=created.run_id,
                params=outcome.parameters,
            )
            await _experiment_call(
                session,
                request.tracking_uri,
                "log_metrics",
                run_id=created.run_id,
                metrics={
                    **outcome.metrics,
                    "training_time_ms": float(outcome.training_time_ms),
                },
                finish_run=True,
            )

        metric_best = await _experiment_call(
            session,
            request.tracking_uri,
            "query_best_run",
            experiment_name=experiment_name,
            metric_name=primary_metric,
            maximize=maximize,
            candidate_run_ids=candidate_run_ids,
        )
        if metric_best.run_id not in outcome_by_run_id:
            raise ModelSelectionError("MLflow selected a run outside this candidate batch")

        candidates_info = [
            {
                "name": outcome.name,
                "metric_name": primary_metric,
                "metric_value": outcome.metrics[primary_metric],
                "training_time_ms": outcome.training_time_ms,
                "parameters": outcome.parameters,
            }
            for outcome in outcomes
        ]
        choice = choose_model(
            candidates=candidates_info,
            task_type=request.task_type.value,
            primary_metric=primary_metric,
            maximize=maximize,
            prediction_goal=request.prediction_goal,
            retry_feedback=request.retry_feedback,
        )
        selected_run_id = run_id_by_name.get(choice.selected_model_name)
        if selected_run_id is None:
            raise ModelSelectionError("agentic decision selected an unknown candidate")
        best = created_by_run_id[selected_run_id]
        best_outcome = outcome_by_run_id[selected_run_id]

        final_pipeline = clone(best_outcome.pipeline)
        final_pipeline.fit(features, target)
        model_dir = artifact_dir.resolve() / str(request.data_prep_result.run_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = model_dir / f"{best.run_id}.joblib"
        joblib.dump(final_pipeline, local_model_path)

        client = MlflowClient(
            tracking_uri=request.tracking_uri,
            registry_uri=request.tracking_uri,
        )
        client.log_artifact(best.run_id, str(local_model_path), artifact_path="model")
        held_out_path = _path_from_file_uri(
            request.data_prep_result.test_artifact_uri
        )
        client.log_artifact(
            best.run_id,
            str(held_out_path),
            artifact_path="evaluation",
        )
        client.log_dict(
            best.run_id,
            {
                "run_id": str(request.data_prep_result.run_id),
                "task_type": request.task_type.value,
                "target_column": request.data_prep_result.target_column,
                "feature_columns": request.data_prep_result.feature_columns,
                "test_artifact_file": held_out_path.name,
                "test_row_count": request.data_prep_result.test_row_count,
            },
            "evaluation/metadata.json",
        )
        registered_name = request.registered_model_name or (
            f"autonomous-{request.task_type.value}-{request.data_prep_result.run_id.hex}"
        )
        registered = await _experiment_call(
            session,
            request.tracking_uri,
            "register_model",
            run_id=best.run_id,
            artifact_path="model",
            registered_model_name=registered_name,
        )

    return ModelSelectionResult(
        run_id=request.data_prep_result.run_id,
        candidate_run_ids=candidate_run_ids,
        best_run_id=best.run_id,
        best_model_uri=(
            f"models:/{registered.registered_model_name}/{registered.model_version}"
        ),
        best_parameters=best_outcome.parameters,
        training_metrics=best_outcome.metrics,
        llm_reasoning=(
            f"Search plan: {search_reasoning}\n\n"
            f"Final model decision: {choice.reasoning}"
        ),
    )


def _run_coroutine(coroutine: Coroutine[Any, Any, _T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, asyncio.run, coroutine).result()


def select_models(
    request: ModelSelectionRequest | dict[str, Any] | str,
    *,
    artifact_dir: str | Path = DEFAULT_MODEL_ARTIFACT_DIR,
) -> ModelSelectionResult:
    """Train, compare, track, and register exactly three candidate models."""

    if isinstance(request, str):
        try:
            validated_request = ModelSelectionRequest.model_validate_json(request)
        except ValueError as exc:
            raise ModelSelectionError(
                "A2A task message must contain valid ModelSelectionRequest JSON"
            ) from exc
    else:
        try:
            validated_request = ModelSelectionRequest.model_validate(request)
        except ValueError as exc:
            raise ModelSelectionError("invalid model-selection request") from exc

    outcomes, features, target, search_reasoning = _train_candidates(
        validated_request
    )
    return _run_coroutine(
        log_and_query_experiment(
            validated_request,
            outcomes,
            features,
            target,
            Path(artifact_dir),
            search_reasoning,
        )
    )


def parse_task_message(message: str) -> ModelSelectionRequest:
    """Parse an A2A text part without accepting prose or untyped fields."""

    try:
        payload = json.loads(message)
        return ModelSelectionRequest.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ModelSelectionError(
            "A2A task message must contain valid ModelSelectionRequest JSON"
        ) from exc


class ModelSelectionAgent(BaseAgent):
    """Google ADK agent that plans and executes bounded model selection."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        message_parts = []
        if ctx.user_content and ctx.user_content.parts:
            message_parts = [part.text for part in ctx.user_content.parts if part.text]
        if not message_parts:
            raise ModelSelectionError("A2A task message must include a JSON text part")

        request = parse_task_message("".join(message_parts))
        result = await asyncio.to_thread(select_models, request)
        yield Event(
            author=self.name,
            branch=ctx.branch,
            invocation_id=ctx.invocation_id,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=result.model_dump_json())],
            ),
        )


root_agent = ModelSelectionAgent(
    name="model_selection_agent",
    description=(
        "Receives a typed DataPrepResult in an A2A JSON task, trains exactly "
        "three task-appropriate candidates, tracks them in MLflow, and returns "
        "a validated ModelSelectionResult."
    ),
)

# ``to_a2a`` supplies the A2A agent card, task store, executor, and JSON-RPC
# routes. Uvicorn binds this ASGI application in ``main`` below.
app = to_a2a(root_agent, host=A2A_HOST, port=A2A_PORT)


async def _send_compatibility_task(request: Request) -> JSONResponse:
    """Accept the Day 2 task endpoint while retaining native A2A JSON-RPC."""

    with extracted_trace_context(dict(request.headers)):
        try:
            payload = await request.json()
            task_payload = (
                payload.get("task", payload) if isinstance(payload, dict) else payload
            )
            validated_request = ModelSelectionRequest.model_validate(task_payload)
            result = await asyncio.to_thread(select_models, validated_request)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return JSONResponse(
                {"error": "invalid model-selection task", "detail": str(exc)},
                status_code=422,
            )
        return JSONResponse(result.model_dump(mode="json"))


app.add_route(
    "/a2a/tasks/send",
    _send_compatibility_task,
    methods=["POST"],
    name="day2-task-send",
)


def main() -> None:
    """Serve the Google ADK agent as a localhost A2A service."""

    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)


if __name__ == "__main__":
    main()
