"""MCP tool for MLflow experiment tracking and model registration.

Run this module directly to serve the tool over MCP's stdio transport::

    python -m tools.log_experiment
"""

from __future__ import annotations

import math
import os
import re
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ParameterValue = str | int | float | bool
_METRIC_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ExperimentAction(StrEnum):
    CREATE_RUN = "create_run"
    LOG_PARAMS = "log_params"
    LOG_METRICS = "log_metrics"
    REGISTER_MODEL = "register_model"
    QUERY_BEST_RUN = "query_best_run"


class ExperimentResult(BaseModel):
    """Validated result returned for every MLflow tool action."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    action: ExperimentAction
    experiment_id: NonBlankString
    run_id: NonBlankString
    status: NonBlankString
    params: dict[NonBlankString, str] = Field(default_factory=dict)
    metrics: dict[NonBlankString, float] = Field(default_factory=dict)
    artifact_uri: NonBlankString | None = None
    model_uri: NonBlankString | None = None
    registered_model_name: NonBlankString | None = None
    model_version: NonBlankString | None = None

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError("metric values must be finite")
        return value

    @model_validator(mode="after")
    def validate_action_result(self) -> "ExperimentResult":
        registration_fields = (
            self.model_uri,
            self.registered_model_name,
            self.model_version,
        )
        if self.action == ExperimentAction.REGISTER_MODEL:
            if any(value is None for value in registration_fields):
                raise ValueError("register_model requires complete model-version metadata")
        elif any(value is not None for value in registration_fields):
            raise ValueError("model-version metadata is only valid for register_model")
        return self


class ExperimentToolError(ValueError):
    """Safe error raised when an MLflow action cannot be completed."""


def _validate_tracking_uri(tracking_uri: str) -> str:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"file", "http", "https", "sqlite"}:
        raise ExperimentToolError(
            "tracking_uri must use the file, sqlite, http, or https scheme"
        )
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ExperimentToolError("HTTP tracking_uri must include a host")
    if parsed.scheme in {"file", "sqlite"} and not parsed.path:
        raise ExperimentToolError("local tracking_uri must include a path")
    return tracking_uri


def _validate_run_id(run_id: str | None) -> str:
    if run_id is None or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ExperimentToolError("a valid run_id is required for this action")
    return run_id


def _validate_metric_values(metrics: dict[str, float] | None) -> dict[str, float]:
    if not metrics:
        raise ExperimentToolError("at least one metric is required")
    if any(not name.strip() or not _METRIC_NAME_PATTERN.fullmatch(name) for name in metrics):
        raise ExperimentToolError(
            "metric names may contain only letters, numbers, dots, underscores, and hyphens"
        )
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ExperimentToolError("metric values must be finite")
    return metrics


def _validate_artifact_path(artifact_path: str | None) -> str:
    if artifact_path is None or not artifact_path.strip():
        raise ExperimentToolError("artifact_path is required for register_model")
    normalized = PurePosixPath(artifact_path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or str(normalized) == ".":
        raise ExperimentToolError("artifact_path must be a safe run-relative path")
    return str(normalized)


def _experiment_id(client: MlflowClient, experiment_name: str, *, create: bool) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    if not create:
        raise ExperimentToolError("MLflow experiment does not exist")
    return client.create_experiment(experiment_name)


def _result_from_run(action: ExperimentAction, run: Run) -> ExperimentResult:
    return ExperimentResult(
        action=action,
        experiment_id=run.info.experiment_id,
        run_id=run.info.run_id,
        status=run.info.status,
        params=dict(run.data.params),
        metrics={name: float(value) for name, value in run.data.metrics.items()},
        artifact_uri=run.info.artifact_uri,
    )


def _create_run(
    client: MlflowClient,
    experiment_name: str | None,
    run_name: str | None,
    tags: dict[str, str] | None,
) -> ExperimentResult:
    if experiment_name is None or not experiment_name.strip():
        raise ExperimentToolError("experiment_name is required for create_run")
    experiment_id = _experiment_id(client, experiment_name, create=True)
    run = client.create_run(experiment_id, run_name=run_name, tags=tags)
    return _result_from_run(ExperimentAction.CREATE_RUN, run)


def _log_params(
    client: MlflowClient,
    run_id: str | None,
    params: dict[str, ParameterValue] | None,
) -> ExperimentResult:
    valid_run_id = _validate_run_id(run_id)
    if not params:
        raise ExperimentToolError("at least one parameter is required")
    if any(not name.strip() for name in params):
        raise ExperimentToolError("parameter names must not be blank")
    for name, value in params.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ExperimentToolError("floating-point parameter values must be finite")
        client.log_param(valid_run_id, name, value)
    return _result_from_run(
        ExperimentAction.LOG_PARAMS, client.get_run(valid_run_id)
    )


def _log_metrics(
    client: MlflowClient,
    run_id: str | None,
    metrics: dict[str, float] | None,
    step: int,
    finish_run: bool,
) -> ExperimentResult:
    valid_run_id = _validate_run_id(run_id)
    valid_metrics = _validate_metric_values(metrics)
    if step < 0:
        raise ExperimentToolError("step cannot be negative")
    for name, value in valid_metrics.items():
        client.log_metric(valid_run_id, name, value, step=step)
    if finish_run:
        client.set_terminated(valid_run_id, status="FINISHED")
    return _result_from_run(
        ExperimentAction.LOG_METRICS, client.get_run(valid_run_id)
    )


def _register_model(
    client: MlflowClient,
    run_id: str | None,
    artifact_path: str | None,
    registered_model_name: str | None,
) -> ExperimentResult:
    valid_run_id = _validate_run_id(run_id)
    valid_artifact_path = _validate_artifact_path(artifact_path)
    if registered_model_name is None or not registered_model_name.strip():
        raise ExperimentToolError(
            "registered_model_name is required for register_model"
        )

    run = client.get_run(valid_run_id)
    if not client.list_artifacts(valid_run_id, valid_artifact_path):
        raise ExperimentToolError("artifact_path does not exist in the MLflow run")

    try:
        client.get_registered_model(registered_model_name)
    except MlflowException:
        client.create_registered_model(registered_model_name)

    model_uri = f"runs:/{valid_run_id}/{valid_artifact_path}"
    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=valid_run_id,
        await_creation_for=10,
    )
    return ExperimentResult(
        action=ExperimentAction.REGISTER_MODEL,
        experiment_id=run.info.experiment_id,
        run_id=valid_run_id,
        status=model_version.status or "READY",
        params=dict(run.data.params),
        metrics={name: float(value) for name, value in run.data.metrics.items()},
        artifact_uri=run.info.artifact_uri,
        model_uri=model_uri,
        registered_model_name=model_version.name,
        model_version=str(model_version.version),
    )


def _query_best_run(
    client: MlflowClient,
    experiment_name: str | None,
    metric_name: str | None,
    maximize: bool,
    candidate_run_ids: list[str] | None,
) -> ExperimentResult:
    if experiment_name is None or not experiment_name.strip():
        raise ExperimentToolError("experiment_name is required for query_best_run")
    if metric_name is None or not _METRIC_NAME_PATTERN.fullmatch(metric_name):
        raise ExperimentToolError("a valid metric_name is required for query_best_run")

    experiment_id = _experiment_id(client, experiment_name, create=False)
    direction = "DESC" if maximize else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=1000,
        order_by=[f"metrics.{metric_name} {direction}"],
    )
    allowed_run_ids = None
    if candidate_run_ids is not None:
        if not candidate_run_ids:
            raise ExperimentToolError("candidate_run_ids cannot be empty")
        allowed_run_ids = {_validate_run_id(run_id) for run_id in candidate_run_ids}
    best_run = next(
        (
            run
            for run in runs
            if metric_name in run.data.metrics
            and (allowed_run_ids is None or run.info.run_id in allowed_run_ids)
        ),
        None,
    )
    if best_run is None:
        raise ExperimentToolError("no run has logged the requested metric")
    return _result_from_run(ExperimentAction.QUERY_BEST_RUN, best_run)


def log_experiment(
    action: ExperimentAction,
    tracking_uri: Annotated[
        str,
        Field(description="MLflow tracking and registry URI."),
    ],
    experiment_name: str | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
    params: dict[str, ParameterValue] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    step: int = 0,
    finish_run: bool = False,
    artifact_path: str | None = None,
    registered_model_name: str | None = None,
    metric_name: str | None = None,
    maximize: bool = True,
    candidate_run_ids: list[str] | None = None,
) -> ExperimentResult:
    """Execute one validated MLflow tracking or registry action."""

    try:
        action = ExperimentAction(action)
    except ValueError as exc:
        raise ExperimentToolError("unsupported experiment action") from exc
    valid_tracking_uri = _validate_tracking_uri(tracking_uri)
    try:
        client = MlflowClient(
            tracking_uri=valid_tracking_uri,
            registry_uri=valid_tracking_uri,
        )
        if action == ExperimentAction.CREATE_RUN:
            return _create_run(client, experiment_name, run_name, tags)
        if action == ExperimentAction.LOG_PARAMS:
            return _log_params(client, run_id, params)
        if action == ExperimentAction.LOG_METRICS:
            return _log_metrics(client, run_id, metrics, step, finish_run)
        if action == ExperimentAction.REGISTER_MODEL:
            return _register_model(
                client,
                run_id,
                artifact_path,
                registered_model_name,
            )
        if action == ExperimentAction.QUERY_BEST_RUN:
            return _query_best_run(
                client,
                experiment_name,
                metric_name,
                maximize,
                candidate_run_ids,
            )
    except ExperimentToolError:
        raise
    except MlflowException as exc:
        raise ExperimentToolError(
            f"MLflow could not complete the {action.value} action"
        ) from exc
    except OSError as exc:
        raise ExperimentToolError("MLflow storage could not be accessed") from exc

    raise ExperimentToolError("unsupported experiment action")


def _default_tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    database = (Path("/tmp/artifacts") / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def log_and_query_experiment(
    experiment_name: str,
    *,
    params: dict[str, ParameterValue] | None = None,
    metrics: dict[str, float] | None = None,
    tracking_uri: str | None = None,
    run_name: str | None = None,
) -> ExperimentResult:
    """Create one run, log supplied values, and return its final state."""

    effective_uri = tracking_uri or _default_tracking_uri()
    created = log_experiment(
        "create_run",
        effective_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    if params:
        log_experiment(
            "log_params",
            effective_uri,
            run_id=created.run_id,
            params=params,
        )
    if metrics:
        return log_experiment(
            "log_metrics",
            effective_uri,
            run_id=created.run_id,
            metrics=metrics,
            finish_run=True,
        )
    return created


def query_best_run(
    experiment_name: str,
    *,
    metric: str,
    maximize: bool = True,
    tracking_uri: str | None = None,
) -> ExperimentResult:
    """Return the best run for a named metric from the configured store."""

    return log_experiment(
        "query_best_run",
        tracking_uri or _default_tracking_uri(),
        experiment_name=experiment_name,
        metric_name=metric,
        maximize=maximize,
    )


server = FastMCP("mlflow-experiment")
server.tool(
    name="log_experiment",
    title="Manage MLflow experiment",
    description=(
        "Create MLflow runs, log parameters and metrics, register a model, or "
        "query the best run using one typed action interface."
    ),
)(log_experiment)

# Keep the original concise tool name for direct clients while exposing the
# orchestration-facing name used by the model-selection agent. Both names map
# to the same typed action interface and therefore the same MCP capability.
server.tool(
    name="log_and_query_experiment",
    title="Log and query MLflow experiment",
    description=(
        "Create MLflow runs, log parameters and metrics, register a model, or "
        "query the best run using one typed action interface."
    ),
)(log_experiment)


def main() -> None:
    """Run the MCP server over stdio."""

    server.run()


if __name__ == "__main__":
    main()
