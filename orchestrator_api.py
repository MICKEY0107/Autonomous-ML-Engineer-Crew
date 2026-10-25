"""FastAPI facade for the Autonomous ML Engineer Crew dashboard.

The API intentionally exposes only schemas, metrics, artifact references, and
sanitized trace metadata. Dataset rows never cross this boundary.
"""

from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from contracts.models import PipelineState, PipelineStatus
from main import _pipeline_result, _start_a2a_server, create_pipeline_state, infer_task_type
from orchestrator.flow import DEFAULT_A2A_URL, FlowCallbackEvent, FlowCallbacks, build_flow
from orchestrator.telemetry import (
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_OUTPUT,
    flush_traces,
    get_tracer_provider,
    langfuse_trace_url,
    redact_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parent
UI_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "ui"
UI_MLFLOW_DATABASE = (UI_ARTIFACT_ROOT / "mlflow.db").resolve()
UI_DIST = PROJECT_ROOT / "ui" / "dist"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MLFLOW_URL = os.environ["MLFLOW_UI_URL"]
API_CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.environ["API_CORS_ORIGINS"].split(",")
    if origin.strip()
)

STAGE_NAMES = (
    "Input Validation",
    "Data-Prep Agent",
    "MCP: profile_dataset",
    "Model-Selection Agent",
    "A2A: tasks.send",
    "MCP: log_experiment",
    "Evaluation Agent",
    "Retry Controller",
    "Terminal Result",
)

SPAN_STAGE_NAMES = {
    "agent.data_prep": "Data-Prep Agent",
    "mcp.profile_dataset": "MCP: profile_dataset",
    "agent.model_selection": "Model-Selection Agent",
    "a2a.tasks.send": "A2A: tasks.send",
    "mcp.log_and_query_experiment": "MCP: log_experiment",
    "agent.evaluation": "Evaluation Agent",
    "pipeline.run": "Terminal Result",
}


def _fresh_stages() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "duration_ms": 0,
            "status": "pending",
            "detail": "Awaiting pipeline run",
            "attempts": 0,
            "reasoning": {},
            "_started_at": None,
        }
        for name in STAGE_NAMES
    ]


_lock = threading.RLock()
_state: dict[str, Any] = {
    "status": "idle",
    "current_stage": None,
    "retry_count": 0,
    "stages_completed": _fresh_stages(),
    "final_result": None,
    "trace_data": None,
    "llm_insights": None,
    "error": None,
    "started_at": None,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
_worker: threading.Thread | None = None


def _snapshot() -> dict[str, Any]:
    with _lock:
        snapshot = deepcopy(_state)
    for item in snapshot.get("stages_completed", []):
        started_at = item.pop("_started_at", None)
        if item.get("status") == "running" and started_at is not None:
            item["duration_ms"] = max(
                int(item.get("duration_ms", 0)),
                int((time.perf_counter() - started_at) * 1_000),
            )
    snapshot["stages"] = deepcopy(snapshot.get("stages_completed", []))
    return snapshot


def _touch() -> None:
    _state["updated_at"] = datetime.now(timezone.utc).isoformat()


def _stage(name: str) -> dict[str, Any]:
    return next(item for item in _state["stages_completed"] if item["name"] == name)


def _sanitize_reasoning(value: Any) -> Any:
    """Redact row-bearing keys while preserving safe structured explanations."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in {
                "raw_rows",
                "dataset_rows",
                "records",
                "samples",
                "train_data",
                "test_data",
                "training_data",
                "dataframe",
            }:
                sanitized[str(key)] = "[REDACTED_RAW_ROW_DATA]"
            else:
                sanitized[str(key)] = _sanitize_reasoning(child)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_reasoning(item) for item in value]
    return redact_payload(value)


def _set_stage(
    name: str,
    status: str,
    *,
    detail: str | None = None,
    duration_ms: int | None = None,
    increment_attempt: bool = False,
    reasoning: dict[str, Any] | None = None,
) -> None:
    with _lock:
        item = _stage(name)
        item["status"] = status
        if detail is not None:
            item["detail"] = detail
        if duration_ms is not None:
            item["duration_ms"] = max(0, int(duration_ms))
        elif status != "running" and item.get("_started_at") is not None:
            item["duration_ms"] = max(
                int(item.get("duration_ms", 0)),
                int((time.perf_counter() - item["_started_at"]) * 1_000),
            )
        if increment_attempt:
            item["attempts"] += 1
        if reasoning is not None:
            item["reasoning"] = _sanitize_reasoning(reasoning)
        if status == "running":
            if item.get("_started_at") is None:
                item["_started_at"] = time.perf_counter()
            _state["current_stage"] = name
        else:
            item["_started_at"] = None
        _touch()


def _safe_json_attribute(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return redact_payload(value)
    try:
        return redact_payload(json.loads(value))
    except json.JSONDecodeError:
        return value[:2_000]


class _DashboardSpanExporter(SpanExporter):
    """Mirror completed row-free spans into the live dashboard state."""

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            attributes = dict(span.attributes or {})
            run_id = str(attributes.get("ml.run_id", ""))
            with _lock:
                active_run = str((_state.get("final_result") or {}).get("run_id", ""))
                pending_run = str(_state.get("run_id", ""))
            if run_id and run_id not in {active_run, pending_run}:
                continue
            duration_ms = int(((span.end_time or 0) - (span.start_time or 0)) / 1_000_000)
            stage_name = SPAN_STAGE_NAMES.get(span.name)
            if stage_name:
                span_input = _safe_json_attribute(
                    attributes.get(LANGFUSE_OBSERVATION_INPUT)
                )
                span_output = _safe_json_attribute(
                    attributes.get(LANGFUSE_OBSERVATION_OUTPUT)
                )
                _set_stage(
                    stage_name,
                    "success",
                    duration_ms=duration_ms,
                    detail=f"Completed in {duration_ms / 1000:.2f}s",
                    reasoning={
                        key: value
                        for key, value in {
                            "input": span_input,
                            "output": span_output,
                        }.items()
                        if value is not None
                    },
                )
            node = {
                "name": span.name,
                "duration_ms": duration_ms,
                "input": _safe_json_attribute(attributes.get(LANGFUSE_OBSERVATION_INPUT)),
                "output": _safe_json_attribute(attributes.get(LANGFUSE_OBSERVATION_OUTPUT)),
                "status": "error" if span.status.status_code.name == "ERROR" else "success",
            }
            with _lock:
                _state.setdefault("captured_spans", []).append(node)
                _touch()
        return SpanExportResult.SUCCESS


get_tracer_provider().add_span_processor(SimpleSpanProcessor(_DashboardSpanExporter()))


def _callbacks() -> FlowCallbacks:
    stage_map = {
        "data_prep": "Data-Prep Agent",
        "model_selection": "Model-Selection Agent",
        "evaluation": "Evaluation Agent",
    }

    def on_start(event: FlowCallbackEvent) -> None:
        stage_name = stage_map.get(event.stage)
        if stage_name:
            _set_stage(
                stage_name,
                "running",
                detail=f"Attempt {event.retry_count + 1} in progress",
                increment_attempt=True,
            )
            if event.stage == "data_prep":
                _set_stage("MCP: profile_dataset", "running", detail="Profiling dataset")
            elif event.stage == "model_selection":
                _set_stage("A2A: tasks.send", "running", detail="Dispatching ADK task")
                _set_stage("MCP: log_experiment", "running", detail="Tracking candidate runs")

    def on_complete(event: FlowCallbackEvent) -> None:
        stage_name = stage_map.get(event.stage)
        if stage_name:
            _set_stage(stage_name, "success", detail=f"Attempt {event.retry_count + 1} complete")
        if event.stage == "data_prep":
            _set_stage("MCP: profile_dataset", "success", detail="Schema and quality profile captured")
        elif event.stage == "model_selection":
            _set_stage("A2A: tasks.send", "success", detail="Validated ADK response")
            _set_stage("MCP: log_experiment", "success", detail="MLflow experiment recorded")
        elif event.stage == "evaluation" and event.retry_count:
            _set_stage("Retry Controller", "success", detail=f"Retry {event.retry_count}/2 completed")

    def on_error(event: FlowCallbackEvent) -> None:
        stage_name = stage_map.get(event.stage, event.stage)
        if stage_name in STAGE_NAMES:
            _set_stage(stage_name, "failed", detail=event.message or "Stage failed")

    def on_retry(event: FlowCallbackEvent) -> None:
        with _lock:
            _state["retry_count"] = event.retry_count
            retry_history = _state.setdefault("retry_history", [])
            retry_history.append(
                {
                    "attempt": event.retry_count,
                    "status": "failed",
                    "reason": event.message or "Evaluation gate was not met",
                    "action": "Model selection rerun with evaluation feedback",
                }
            )
        _set_stage(
            "Retry Controller",
            "retry",
            detail=f"Re-routing model selection · retry {event.retry_count}/2",
            increment_attempt=True,
            reasoning={
                "retry_count": event.retry_count,
                "max_retries": 2,
                "retry_history": deepcopy(_state.get("retry_history", [])),
                "decision": event.message or "Evaluation gate was not met",
            },
        )

    def on_validation_failure(event: FlowCallbackEvent) -> None:
        on_error(event)

    return FlowCallbacks(
        on_stage_start=on_start,
        on_stage_complete=on_complete,
        on_stage_error=on_error,
        on_retry=on_retry,
        on_validation_failure=on_validation_failure,
    )


def _metric_payload(state: PipelineState) -> tuple[str | None, float | None, float | None]:
    evaluation = state.evaluation_result
    if evaluation is None:
        return None, None, None
    name = evaluation.primary_metric
    return name, float(evaluation.held_out_metrics[name]), float(evaluation.acceptance_threshold)


def _captured_span_output(name: str) -> dict[str, Any]:
    """Return the latest sanitized output for a completed telemetry span."""

    with _lock:
        matches = [
            span.get("output")
            for span in _state.get("captured_spans", [])
            if span.get("name") == name and isinstance(span.get("output"), dict)
        ]
    return deepcopy(matches[-1]) if matches else {}


def _candidate_reasoning(state: PipelineState) -> dict[str, Any]:
    """Build a row-free candidate comparison from this run's MLflow records."""

    selection = state.model_selection_result
    if selection is None:
        return {}
    metric_name = "f1_weighted" if str(state.task_type) == "classification" else "rmse"
    maximize = str(state.task_type) == "classification"
    experiment_name = f"autonomous-model-selection-{state.run_id.hex}"
    candidates: list[dict[str, Any]] = []
    try:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=f"sqlite:///{UI_MLFLOW_DATABASE.as_posix()}")
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            [experiment.experiment_id] if experiment is not None else [],
            order_by=["attributes.start_time ASC"],
            max_results=100,
        ) if experiment is not None else []
        for run in runs:
            model_name = run.data.params.get("model_type")
            if not model_name or metric_name not in run.data.metrics:
                continue
            attempt = int(run.data.tags.get("retry_attempt", "0"))
            candidates.append(
                {
                    "name": model_name,
                    "attempt": attempt + 1,
                    "metric_name": metric_name,
                    "metric_value": float(run.data.metrics[metric_name]),
                    "training_time_ms": int(run.data.metrics.get("training_time_ms", 0)),
                    "selected": run.info.run_id == selection.best_run_id,
                    "run_id": run.info.run_id,
                    "parameters": {
                        key: value
                        for key, value in run.data.params.items()
                        if key != "model_type"
                    },
                }
            )
    except Exception:
        # The selection contract still contains enough verified data to explain
        # the winner if the local MLflow store is temporarily unavailable.
        candidates = []

    if not candidates:
        candidates = [
            {
                "name": selection.best_model_name,
                "attempt": state.retry_count + 1,
                "metric_name": metric_name,
                "metric_value": float(selection.training_metrics[metric_name]),
                "selected": True,
                "run_id": selection.best_run_id,
                "parameters": selection.best_parameters,
            }
        ]

    latest_attempt = max(candidate["attempt"] for candidate in candidates)
    current = [candidate for candidate in candidates if candidate["attempt"] == latest_attempt]
    winner = next((candidate for candidate in current if candidate["selected"]), None)
    if winner is None:
        chooser = max if maximize else min
        winner = chooser(current, key=lambda item: item["metric_value"])
    selection_reason = (
        f"The agentic policy selected {winner['name']} from candidates that passed "
        f"the deterministic {metric_name} quality band in attempt {latest_attempt}. "
        f"Its measured {metric_name} was {winner['metric_value']:.4f}."
    )
    trade_offs = []
    for candidate in current:
        if candidate["run_id"] == winner["run_id"]:
            continue
        delta = abs(float(candidate["metric_value"]) - float(winner["metric_value"]))
        trade_offs.append(
            f"{candidate['name']} differed from the selected model by {delta:.4f} "
            f"on {metric_name}."
        )

    adjustments: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        histories.setdefault(str(candidate["name"]), []).append(candidate)
    for model_name, history in histories.items():
        history.sort(key=lambda item: int(item["attempt"]))
        for previous, current_candidate in zip(history, history[1:]):
            changed = {
                key: {"from": previous["parameters"].get(key), "to": value}
                for key, value in current_candidate["parameters"].items()
                if previous["parameters"].get(key) != value
            }
            if changed:
                adjustments.append(
                    {
                        "model": model_name,
                        "attempt": current_candidate["attempt"],
                        "changes": changed,
                    }
                )

    return {
        "task_type": str(state.task_type),
        "primary_metric": metric_name,
        "metric_direction": "maximize" if maximize else "minimize",
        "candidates": current,
        "candidate_history": candidates,
        "selected_model": winner["name"],
        "hyperparameters": selection.best_parameters,
        "selection_reason": selection_reason,
        "trade_offs": trade_offs,
        "parameter_adjustments": adjustments,
    }


def _apply_state_reasoning(state: PipelineState, final_result: dict[str, Any]) -> None:
    """Attach dynamic, row-free explanations to every dashboard stage."""

    prep = state.data_prep_result
    selection = state.model_selection_result
    evaluation = state.evaluation_result
    if prep is not None:
        prepared_rows = prep.train_row_count + prep.test_row_count
        retained_percent = prepared_rows / prep.row_count * 100
        _set_stage(
            "Input Validation",
            _stage("Input Validation")["status"],
            reasoning={
                "task_type": str(state.task_type),
                "target_column": state.target_column,
                "source_rows": prep.row_count,
                "feature_count": len(prep.feature_columns),
                "retained_percent": round(retained_percent, 2),
                "checks_passed": [
                    "CSV guardrails",
                    "target column validation",
                    "prediction goal injection scan",
                ],
            },
        )
        _set_stage(
            "Data-Prep Agent",
            _stage("Data-Prep Agent")["status"],
            reasoning={
                "rows_before": prep.row_count,
                "rows_after": prepared_rows,
                "rows_removed": prep.row_count - prepared_rows,
                "train_rows": prep.train_row_count,
                "test_rows": prep.test_row_count,
                "feature_count": len(prep.feature_columns),
                "target_column": prep.target_column,
                "operations": prep.quality_findings,
                "train_path": prep.train_artifact_uri,
                "test_path": prep.test_artifact_uri,
                "llm_reasoning": prep.llm_reasoning,
            },
        )
        profile_output = _captured_span_output("mcp.profile_dataset")
        _set_stage(
            "MCP: profile_dataset",
            _stage("MCP: profile_dataset")["status"],
            reasoning={
                "sha256": profile_output.get("sha256", prep.dataset_hash),
                "null_count": profile_output.get("null_count"),
                "feature_count": profile_output.get("feature_count", len(prep.schema_summary)),
                "schema": prep.schema_summary,
            },
        )
    if selection is not None:
        model_reasoning = _candidate_reasoning(state)
        model_reasoning["llm_reasoning"] = selection.llm_reasoning
        _set_stage(
            "Model-Selection Agent",
            _stage("Model-Selection Agent")["status"],
            reasoning=model_reasoning,
        )
        primary_metric = model_reasoning.get("primary_metric")
        training_metric = selection.training_metrics.get(str(primary_metric))
        _set_stage(
            "A2A: tasks.send",
            _stage("A2A: tasks.send")["status"],
            reasoning={
                "task_type": str(state.task_type),
                "retry_count": state.retry_count,
                "best_model_name": selection.best_model_name,
                "metric_name": primary_metric,
                "metric_value": training_metric,
                "validated_response": True,
            },
        )
        _set_stage(
            "MCP: log_experiment",
            _stage("MCP: log_experiment")["status"],
            reasoning={
                "experiment_name": f"autonomous-model-selection-{state.run_id.hex}",
                "candidate_run_count": len(model_reasoning.get("candidate_history", [])),
                "best_run_id": selection.best_run_id,
                "best_metric": {primary_metric: training_metric},
                "parameters": selection.best_parameters,
            },
        )
    if evaluation is not None:
        metrics = [
            {
                "name": name,
                "value": value,
                "threshold": evaluation.acceptance_threshold
                if name == evaluation.primary_metric
                else None,
                "direction": str(evaluation.metric_direction)
                if name == evaluation.primary_metric
                else None,
                "passed": evaluation.passed if name == evaluation.primary_metric else None,
            }
            for name, value in evaluation.held_out_metrics.items()
        ]
        _set_stage(
            "Evaluation Agent",
            _stage("Evaluation Agent")["status"],
            reasoning={
                "test_samples": prep.test_row_count if prep is not None else None,
                "metrics": metrics,
                "primary_metric": evaluation.primary_metric,
                "metric_value": evaluation.held_out_metrics[evaluation.primary_metric],
                "threshold": evaluation.acceptance_threshold,
                "direction": str(evaluation.metric_direction),
                "passed": evaluation.passed,
                "evaluation_reason": evaluation.feedback,
                "threshold_rationale": evaluation.threshold_rationale,
                "llm_reasoning": evaluation.llm_reasoning,
            },
        )
    with _lock:
        retry_history = deepcopy(_state.get("retry_history", []))
    _set_stage(
        "Retry Controller",
        _stage("Retry Controller")["status"],
        reasoning={
            "retry_count": state.retry_count,
            "max_retries": state.max_retries,
            "retry_history": retry_history,
            "decision": "Evaluation passed; no retry needed"
            if state.retry_count == 0 and state.status == PipelineStatus.SUCCESS
            else final_result.get("feedback"),
        },
    )
    _set_stage(
        "Terminal Result",
        _stage("Terminal Result")["status"],
        reasoning={
            "status": final_result["status"],
            "run_id": final_result["run_id"],
            "model_name": final_result.get("model_name"),
            "metric_name": final_result.get("metric_name"),
            "metric_value": final_result.get("metric_value"),
            "retry_count": final_result.get("retry_count"),
            "reason": final_result.get("feedback") or final_result.get("error"),
            "llm_summary": final_result.get("llm_summary"),
        },
    )


def _trace_tree(state: PipelineState) -> dict[str, Any]:
    with _lock:
        spans = deepcopy(_state.get("captured_spans", []))
    by_name: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        by_name.setdefault(span["name"], []).append(span)

    def observations(name: str) -> list[dict[str, Any]]:
        return by_name.get(name, [])

    metric_name, metric_value, _ = _metric_payload(state)
    selection = state.model_selection_result
    prep = state.data_prep_result

    root = (observations("pipeline.run") or [{"name": "pipeline.run", "status": "success"}])[-1]
    root["children"] = []
    for agent_name, child_name in (
        ("agent.data_prep", "mcp.profile_dataset"),
        ("agent.model_selection", "a2a.tasks.send"),
        ("agent.evaluation", ""),
    ):
        agents = observations(agent_name)
        if not agents:
            agents = [{"name": agent_name, "status": "success", "input": None, "output": None}]
        for index, agent in enumerate(agents):
            agent["label"] = f"{agent_name} · attempt {index + 1}" if len(agents) > 1 else agent_name
            agent["children"] = []
            if child_name:
                children = observations(child_name)
                if children:
                    child = children[min(index, len(children) - 1)]
                elif child_name == "mcp.profile_dataset" and prep is not None:
                    child = {
                        "name": child_name,
                        "status": "success",
                        "input": {"csv_path": state.dataset_uri},
                        "output": {
                            "sha256": prep.dataset_hash,
                            "feature_count": len(prep.feature_columns),
                            "row_count": prep.row_count,
                        },
                    }
                elif child_name == "a2a.tasks.send" and selection is not None:
                    child = {
                        "name": child_name,
                        "status": "success",
                        "input": {"task_type": str(state.task_type), "retry_count": index},
                        "output": {
                            "best_model_name": selection.best_model_name,
                            "metric_value": metric_value,
                        },
                    }
                else:
                    child = {
                        "name": child_name,
                        "status": "success",
                        "input": None,
                        "output": None,
                    }
                agent["children"].append(child)
            if agent_name == "agent.model_selection":
                logs = observations("mcp.log_and_query_experiment")
                if logs:
                    agent["children"].append(logs[min(index, len(logs) - 1)])
                elif selection is not None:
                    agent["children"].append(
                        {
                            "name": "mcp.log_and_query_experiment",
                            "status": "success",
                            "input": {
                                "experiment_name": "autonomous-model-selection",
                                "params": selection.best_parameters,
                            },
                            "output": {
                                "run_id": selection.best_run_id,
                                "best_metric": {metric_name: metric_value},
                            },
                        }
                    )
            root["children"].append(agent)
    root["run_id"] = str(state.run_id)
    return root


def _run_job(
    *,
    csv_path: Path,
    target_column: str,
    prediction_goal: str,
    force_fail_eval: bool,
) -> None:
    server = None
    started = time.perf_counter()
    state: PipelineState | None = None
    try:
        validation_started = time.perf_counter()
        _set_stage(
            "Input Validation",
            "running",
            detail="Applying CSV and prompt guardrails",
            reasoning={
                "csv_path": str(csv_path),
                "csv_size_bytes": csv_path.stat().st_size if csv_path.is_file() else None,
                "target_column": target_column,
                "prediction_goal": prediction_goal,
                "checks": [
                    "file size under 100 MB",
                    "more than 10 rows",
                    "safe path",
                    "prompt injection scan",
                    "target column exists",
                ],
            },
        )
        task_type = infer_task_type(csv_path, target_column)
        state = create_pipeline_state(
            csv_path=csv_path,
            target_column=target_column,
            task_type=task_type,
            prediction_goal=prediction_goal,
        )
        with _lock:
            _state["run_id"] = str(state.run_id)
        _set_stage(
            "Input Validation",
            "success",
            duration_ms=int((time.perf_counter() - validation_started) * 1_000),
            detail=f"Accepted {task_type} task",
        )

        artifact_dir = UI_ARTIFACT_ROOT / state.run_id.hex
        UI_MLFLOW_DATABASE.parent.mkdir(parents=True, exist_ok=True)
        server = _start_a2a_server(artifact_dir)
        flow = build_flow(
            state,
            callbacks=_callbacks(),
            a2a_url=DEFAULT_A2A_URL,
            tracking_uri=f"sqlite:///{UI_MLFLOW_DATABASE.as_posix()}",
            artifact_dir=artifact_dir,
            evaluation_threshold=0.99 if force_fail_eval else None,
            suppress_flow_events=True,
        )
        flow.kickoff()
        state = flow.state
        runtime_seconds = time.perf_counter() - started
        result = _pipeline_result(
            state,
            runtime_seconds,
            llm_summary=flow.llm_summary,
        )
        metric_name, metric_value, threshold = _metric_payload(state)
        selection = state.model_selection_result
        terminal_success = state.status == PipelineStatus.SUCCESS
        final_result = {
            "status": str(state.status),
            "run_id": str(state.run_id),
            "model_name": selection.best_model_name if selection else None,
            "metric_value": metric_value,
            "metric_name": metric_name,
            "threshold": threshold,
            "feedback": state.evaluation_result.feedback if state.evaluation_result else None,
            "token_count": result.input_tokens + result.output_tokens,
            "runtime_ms": int(runtime_seconds * 1_000),
            "trace_url": langfuse_trace_url(state.run_id),
            "mlflow_run_id": selection.best_run_id if selection else None,
            "mlflow_url": MLFLOW_URL,
            "retry_count": state.retry_count,
            "error": "; ".join(state.errors) if state.errors else None,
            "completed_at": result.completed_at.isoformat(),
            "llm_summary": result.llm_summary,
        }
        llm_insights = {
            "data_prep": state.data_prep_result.llm_reasoning
            if state.data_prep_result
            else "",
            "model_selection": selection.llm_reasoning if selection else "",
            "evaluation": state.evaluation_result.llm_reasoning
            if state.evaluation_result
            else "",
            "summary": result.llm_summary,
        }
        if _stage("Retry Controller")["status"] == "pending":
            _set_stage(
                "Retry Controller",
                "success",
                detail="No retry required" if state.retry_count == 0 else f"{state.retry_count}/2 retries completed",
            )
        _set_stage(
            "Terminal Result",
            "success" if terminal_success else "failed",
            detail=str(state.status),
            duration_ms=int(runtime_seconds * 1_000),
        )
        _apply_state_reasoning(state, final_result)
        with _lock:
            _state["status"] = "success" if terminal_success else "failed"
            _state["current_stage"] = "Terminal Result"
            _state["retry_count"] = state.retry_count
            _state["final_result"] = final_result
            _state["trace_data"] = _trace_tree(state)
            _state["llm_insights"] = llm_insights
            _state["error"] = final_result["error"]
            _touch()
    except Exception as exc:
        runtime_ms = int((time.perf_counter() - started) * 1_000)
        run_id = str(state.run_id if state is not None else uuid4())
        message = str(redact_payload(exc))[:500]
        if _stage("Input Validation")["status"] == "running":
            _set_stage(
                "Input Validation",
                "failed",
                detail=message,
                duration_ms=runtime_ms,
                reasoning={
                    "csv_path": str(csv_path),
                    "target_column": target_column,
                    "validation_passed": False,
                    "failure_reason": message,
                },
            )
        _set_stage(
            "Terminal Result",
            "failed",
            detail=message,
            duration_ms=runtime_ms,
            reasoning={
                "status": "FAILED_INPUT",
                "run_id": run_id,
                "reason": message,
            },
        )
        with _lock:
            _state["status"] = "failed"
            _state["current_stage"] = "Terminal Result"
            _state["final_result"] = {
                "status": "FAILED_INPUT",
                "run_id": run_id,
                "model_name": None,
                "metric_value": None,
                "metric_name": None,
                "threshold": None,
                "feedback": message,
                "token_count": 0,
                "runtime_ms": runtime_ms,
                "trace_url": langfuse_trace_url(UUID(run_id)),
                "mlflow_run_id": None,
                "mlflow_url": MLFLOW_URL,
                "retry_count": 0,
                "error": message,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "llm_summary": "",
            }
            _state["llm_insights"] = {
                "data_prep": "",
                "model_selection": "",
                "evaluation": "",
                "summary": "",
            }
            _state["error"] = message
            _touch()
    finally:
        if server is not None:
            server.stop()
        flush_traces()


app = FastAPI(
    title="Autonomous ML Engineer Crew API",
    version="1.0.0",
    description="Row-safe asynchronous dashboard API for the existing CrewAI Flow.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(API_CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/pipeline/run", status_code=202)
async def run_pipeline_endpoint(
    csv_file: UploadFile = File(...),
    target_column: str = Form(...),
    prediction_goal: str = Form(...),
    force_fail_eval: bool = Form(False),
) -> dict[str, Any]:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            raise HTTPException(status_code=409, detail="A pipeline run is already active")

    filename = Path(csv_file.filename or "upload.csv").name
    if not filename.casefold().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV uploads are accepted")
    upload_id = uuid4().hex
    upload_dir = UI_ARTIFACT_ROOT / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    destination = upload_dir / filename
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await csv_file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="CSV must be smaller than 100 MB")
                output.write(chunk)
    finally:
        await csv_file.close()

    with _lock:
        _state.clear()
        _state.update(
            {
                "status": "running",
                "current_stage": "Input Validation",
                "retry_count": 0,
                "stages_completed": _fresh_stages(),
                "final_result": None,
                "trace_data": None,
                "llm_insights": None,
                "captured_spans": [],
                "retry_history": [],
                "error": None,
                "run_id": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    _worker = threading.Thread(
        target=_run_job,
        kwargs={
            "csv_path": destination,
            "target_column": target_column,
            "prediction_goal": prediction_goal,
            "force_fail_eval": force_fail_eval,
        },
        name=f"pipeline-{upload_id[:8]}",
        daemon=True,
    )
    _worker.start()
    return _snapshot()


@app.get("/api/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    return _snapshot()


@app.get("/api/pipeline/result")
def pipeline_result() -> dict[str, Any]:
    snapshot = _snapshot()
    if snapshot["status"] in {"idle", "running"} or snapshot["final_result"] is None:
        raise HTTPException(status_code=404, detail="No completed pipeline result is available")
    return {
        "final_result": snapshot["final_result"],
        "trace_data": snapshot["trace_data"],
        "llm_insights": snapshot["llm_insights"],
        "stages_completed": snapshot["stages_completed"],
        "stages": snapshot["stages"],
        "updated_at": snapshot["updated_at"],
    }


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the compiled React dashboard from the same origin as the API."""
    index_file = UI_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=503,
            detail="Dashboard build is unavailable. Run `npm run build` in ui first.",
        )
    return FileResponse(index_file)


@app.get("/favicon.svg", include_in_schema=False)
def dashboard_favicon() -> FileResponse:
    favicon = UI_DIST / "favicon.svg"
    if not favicon.is_file():
        raise HTTPException(status_code=404, detail="Dashboard favicon is unavailable")
    return FileResponse(favicon, media_type="image/svg+xml")


if (UI_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="dashboard-assets")

