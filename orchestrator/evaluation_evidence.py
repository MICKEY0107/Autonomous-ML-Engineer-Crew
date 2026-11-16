"""Build and persist cloud-safe evaluation evidence for dashboard runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from contracts.evaluation import EvaluationRun, EvaluationSuite, EvidenceEvent
from orchestrator.telemetry import redact_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_EVIDENCE_DIR = PROJECT_ROOT / "artifacts" / "evaluation_runs"
GCS_PREFIX = "evaluation-runs"
SUITE_PREFIX = "evaluation-suites"


def _runtime_environment() -> dict[str, Any]:
    return {
        "deployment": "google_cloud_run" if os.getenv("K_SERVICE") else "local",
        "service": os.getenv("K_SERVICE", "local"),
        "revision": os.getenv("K_REVISION", os.getenv("DEPLOYMENT_REVISION", "unknown")),
        "region": os.getenv("CLOUD_RUN_REGION", "unknown"),
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "llm_mode": os.getenv("LLM_MODE", "offline"),
        "model": os.getenv("CREWAI_MODEL", "unconfigured"),
        "llm_cache": os.getenv("LLM_CACHE", "on"),
        "a2a_port": os.getenv("A2A_BIND_PORT", "8081"),
    }


def _cloud_run_estimate(runtime_ms: int | float) -> dict[str, Any]:
    """Calculate a transparent estimate only when current regional rates are configured."""

    try:
        vcpu = float(os.environ["CLOUD_RUN_VCPU"])
        gib = float(os.environ["CLOUD_RUN_MEMORY_GIB"])
        vcpu_rate = float(os.environ["CLOUD_RUN_VCPU_USD_PER_SECOND"])
        gib_rate = float(os.environ["CLOUD_RUN_MEMORY_GIB_USD_PER_SECOND"])
    except (KeyError, ValueError):
        return {"estimated_cloud_run_cost_usd": None, "pricing_configured": False}
    runtime_seconds = max(0.0, float(runtime_ms) / 1_000)
    return {
        "estimated_cloud_run_cost_usd": runtime_seconds * (vcpu * vcpu_rate + gib * gib_rate),
        "pricing_configured": True,
        "runtime_seconds": runtime_seconds,
        "vcpu": vcpu,
        "memory_gib": gib,
        "vcpu_usd_per_second": vcpu_rate,
        "memory_gib_usd_per_second": gib_rate,
    }


def _stage_events(stages: list[dict[str, Any]]) -> list[EvidenceEvent]:
    events: list[EvidenceEvent] = []
    for stage in stages:
        name = str(stage.get("name", "pipeline stage"))
        category = "pipeline"
        if name.startswith("MCP:"):
            category = "mcp"
        elif name.startswith("A2A:"):
            category = "a2a"
        elif "Agent" in name:
            category = "agent"
        status = str(stage.get("status", "failed"))
        if status not in {"success", "failed", "running", "retry", "blocked", "passed"}:
            status = "failed"
        events.append(
            EvidenceEvent(
                category=category,  # type: ignore[arg-type]
                name=name,
                status=status,  # type: ignore[arg-type]
                duration_ms=int(stage.get("duration_ms") or 0),
                detail=str(stage.get("detail") or ""),
                metadata=redact_payload(stage.get("reasoning") or {}),
            )
        )
        if name == "Input Validation":
            events.append(
                EvidenceEvent(
                    category="guardrail",
                    name="CSV and prompt guardrails",
                    status="passed" if status == "success" else "failed",
                    duration_ms=int(stage.get("duration_ms") or 0),
                    detail="CSV schema, target-column, path, size, and prompt-injection validation",
                    metadata={"raw_rows_exported": False},
                )
            )
    return events


def build_evidence(payload: dict[str, Any]) -> EvaluationRun:
    """Translate the existing dashboard payload into one assessment record."""

    final = payload.get("final_result") or {}
    insights = payload.get("insights") or {}
    dataset = insights.get("dataset") or {}
    stages = payload.get("stages_completed") or payload.get("stages") or []
    callbacks = payload.get("callback_events") or []
    a2a = payload.get("a2a") or {}
    events = _stage_events(stages)

    for callback in callbacks:
        status = str(callback.get("status", "success"))
        if status not in {"success", "failed", "running", "retry", "blocked", "passed"}:
            status = "failed"
        events.append(
            EvidenceEvent(
                category="callback",
                name=str(callback.get("name", "flow callback")),
                status=status,  # type: ignore[arg-type]
                detail=str(callback.get("detail") or ""),
                metadata=redact_payload(callback.get("metadata") or {}),
            )
        )

    for exchange in a2a.get("exchanges") or []:
        events.append(
            EvidenceEvent(
                category="a2a",
                name=str(exchange.get("method", "SendMessage")),
                status="success" if str(exchange.get("task_state", "")).endswith("COMPLETED") else "failed",
                duration_ms=int(exchange.get("latency_ms") or 0),
                source="CrewAI Flow",
                destination=str(exchange.get("agent") or "Google ADK agent"),
                detail=str(exchange.get("task_id") or ""),
                metadata=redact_payload(
                    {
                        "protocol_version": exchange.get("protocol_version"),
                        "task_state": exchange.get("task_state"),
                    }
                ),
            )
        )

    quality: dict[str, float] = {}
    primary_metric = final.get("metric_name")
    for stage in stages:
        if stage.get("name") != "Evaluation Agent":
            continue
        for metric in (stage.get("reasoning") or {}).get("metrics") or []:
            name, value = metric.get("name"), metric.get("value")
            if isinstance(name, str) and isinstance(value, (int, float)):
                quality[name] = float(value)

    usage = final.get("usage_report") or {}
    guardrail_events = [event for event in events if event.category == "guardrail"]
    return EvaluationRun(
        run_id=UUID(str(final["run_id"])),
        status=str(final.get("status", "UNKNOWN")),
        environment=_runtime_environment(),
        dataset={
            "name": (payload.get("run_input") or {}).get("dataset_name"),
            "hash": dataset.get("dataset_hash"),
            "rows": dataset.get("rows"),
            "columns": dataset.get("columns"),
            "target_column": dataset.get("target_column") or final.get("target_column"),
            "task_type": dataset.get("task_type") or final.get("task_type"),
            "train_rows": dataset.get("train_rows"),
            "test_rows": dataset.get("test_rows"),
        },
        model_quality=quality,
        primary_metric=primary_metric,
        primary_metric_value=final.get("metric_value"),
        primary_threshold=final.get("threshold"),
        pipeline={
            "runtime_ms": final.get("runtime_ms", 0),
            "retry_count": final.get("retry_count", 0),
            "stage_count": len(stages),
            "successful_stages": sum(1 for stage in stages if stage.get("status") == "success"),
            "failed_stage": final.get("failed_stage"),
        },
        events=events,
        observability={
            "langfuse_trace_url": final.get("trace_url"),
            "mlflow_run_id": final.get("mlflow_run_id"),
            "mlflow_url": final.get("mlflow_url"),
            "cloud_logging_query": f'jsonPayload.run_id="{final.get("run_id", "")}"',
            "a2a_exchange_count": len(a2a.get("exchanges") or []),
        },
        cost={
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", final.get("token_count", 0)),
            "llm_requests": usage.get("llm_requests", 0),
            "llm_latency_seconds": usage.get("llm_latency_seconds", 0),
            "estimated_llm_cost_usd": final.get("estimated_cost_usd", 0.0),
            "pricing": usage.get("pricing") or {},
            "cloud_run": _cloud_run_estimate(final.get("runtime_ms", 0)),
            "cloud_run_cost_note": "Cloud Run estimates exclude any free-tier allowance, network egress, storage, and external observability charges.",
        },
        security={
            "raw_rows_exported": False,
            "guardrail_event_count": len(guardrail_events),
            "validation_callbacks": sum(1 for item in callbacks if "validation" in str(item.get("name", "")).lower()),
        },
    )


def _bucket_name() -> str:
    return os.getenv("EVALUATION_EVIDENCE_BUCKET", "").strip()


def save_evidence(evidence: EvaluationRun) -> dict[str, Any]:
    """Persist locally and, when configured, to private Google Cloud Storage."""

    body = evidence.model_dump_json(indent=2)
    LOCAL_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_EVIDENCE_DIR / f"{evidence.run_id}.json"
    local_path.write_text(body, encoding="utf-8")
    location: dict[str, Any] = {"local_path": str(local_path), "backend": "local"}
    bucket_name = _bucket_name()
    if not bucket_name:
        return location
    try:
        from google.cloud import storage

        blob_name = f"{GCS_PREFIX}/{evidence.run_id}/evidence.json"
        bucket = storage.Client().bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(body, content_type="application/json")
        return {
            "backend": "gcs",
            "bucket": bucket_name,
            "object": blob_name,
            "uri": f"gs://{bucket_name}/{blob_name}",
            "local_path": str(local_path),
        }
    except Exception as exc:  # persistence is recorded but must not fail a completed run
        location["cloud_error"] = type(exc).__name__
        return location


def get_evidence(run_id: str) -> EvaluationRun | None:
    bucket_name = _bucket_name()
    if bucket_name:
        try:
            from google.cloud import storage

            blob = storage.Client().bucket(bucket_name).blob(f"{GCS_PREFIX}/{run_id}/evidence.json")
            if blob.exists():
                return EvaluationRun.model_validate_json(blob.download_as_text())
        except Exception:
            pass
    path = LOCAL_EVIDENCE_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))


def list_evidence(limit: int = 50) -> list[EvaluationRun]:
    bucket_name = _bucket_name()
    records: list[EvaluationRun] = []
    if bucket_name:
        try:
            from google.cloud import storage

            client = storage.Client()
            names = [blob.name for blob in client.list_blobs(bucket_name, prefix=f"{GCS_PREFIX}/") if blob.name.endswith("/evidence.json")]
            for name in sorted(names, reverse=True)[:limit]:
                body = client.bucket(bucket_name).blob(name).download_as_text()
                records.append(EvaluationRun.model_validate_json(body))
            return sorted(records, key=lambda item: item.generated_at, reverse=True)
        except Exception:
            records = []
    for path in sorted(LOCAL_EVIDENCE_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            records.append(EvaluationRun.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(records, key=lambda item: item.generated_at, reverse=True)


def evidence_json(run_id: str) -> str | None:
    record = get_evidence(run_id)
    return record.model_dump_json(indent=2) if record else None


def save_suite(suite: EvaluationSuite) -> dict[str, Any]:
    """Persist formal Cloud Run Job results using the same durable evidence store."""

    body = suite.model_dump_json(indent=2)
    directory = LOCAL_EVIDENCE_DIR / "suites"
    directory.mkdir(parents=True, exist_ok=True)
    local_path = directory / f"{suite.suite_id}.json"
    local_path.write_text(body, encoding="utf-8")
    location: dict[str, Any] = {"backend": "local", "local_path": str(local_path)}
    bucket_name = _bucket_name()
    if not bucket_name:
        return location
    try:
        from google.cloud import storage

        object_name = f"{SUITE_PREFIX}/{suite.suite_id}/suite.json"
        storage.Client().bucket(bucket_name).blob(object_name).upload_from_string(
            body, content_type="application/json"
        )
        return {"backend": "gcs", "uri": f"gs://{bucket_name}/{object_name}", "object": object_name}
    except Exception as exc:
        location["cloud_error"] = type(exc).__name__
        return location


def latest_suite() -> EvaluationSuite | None:
    """Return the newest retained formal suite, preferring Cloud Storage."""

    bucket_name = _bucket_name()
    if bucket_name:
        try:
            from google.cloud import storage

            names = [blob.name for blob in storage.Client().list_blobs(bucket_name, prefix=f"{SUITE_PREFIX}/") if blob.name.endswith("/suite.json")]
            if names:
                body = storage.Client().bucket(bucket_name).blob(sorted(names)[-1]).download_as_text()
                return EvaluationSuite.model_validate_json(body)
        except Exception:
            pass
    directory = LOCAL_EVIDENCE_DIR / "suites"
    candidates = sorted(directory.glob("*.json"), reverse=True) if directory.is_dir() else []
    if not candidates:
        return None
    try:
        return EvaluationSuite.model_validate_json(candidates[0].read_text(encoding="utf-8"))
    except Exception:
        return None
