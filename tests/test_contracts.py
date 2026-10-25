"""Schema and validation tests for the shared pipeline contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from contracts.models import (
    DataPrepResult,
    DeploymentResult,
    EvaluationResult,
    ModelSelectionResult,
    PipelineResult,
    PipelineState,
)


RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
HASH = "a" * 64
IMAGE_DIGEST = f"sha256:{'b' * 64}"


def data_prep() -> DataPrepResult:
    return DataPrepResult(
        run_id=RUN_ID,
        dataset_hash=HASH,
        schema_summary={"age": "float64", "churned": "int64"},
        quality_findings=["No target values are missing"],
        target_column="churned",
        feature_columns=["age"],
        row_count=100,
        train_row_count=80,
        test_row_count=20,
        train_artifact_uri="file:///artifacts/train.parquet",
        test_artifact_uri="file:///artifacts/test.parquet",
    )


def model_selection() -> ModelSelectionResult:
    return ModelSelectionResult(
        run_id=RUN_ID,
        candidate_run_ids=["run-linear", "run-forest"],
        best_run_id="run-forest",
        best_model_uri="mlflow://models/churn/7",
        best_parameters={"n_estimators": 100, "class_weight": "balanced"},
        training_metrics={"f1": 0.86, "roc_auc": 0.91},
    )


def evaluation(*, passed: bool = True) -> EvaluationResult:
    return EvaluationResult(
        run_id=RUN_ID,
        model_uri="mlflow://models/churn/7",
        held_out_metrics={"f1": 0.84, "roc_auc": 0.90},
        primary_metric="f1",
        acceptance_threshold=0.80 if passed else 0.90,
        metric_direction="maximize",
        threshold_rationale="F1 balances false positives and false negatives.",
        passed=passed,
        retry_feedback=None if passed else "Improve minority-class recall.",
    )


def deployment(*, health_status: str = "healthy") -> DeploymentResult:
    return DeploymentResult(
        run_id=RUN_ID,
        image_digest=IMAGE_DIGEST,
        service_url="https://churn.example.run.app",
        health_status=health_status,
        deployment_revision="churn-api-00007",
        deployed_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
    )


def completed_state() -> PipelineState:
    return PipelineState(
        run_id=RUN_ID,
        dataset_uri="file:///datasets/churn.csv",
        task_type="classification",
        target_column="churned",
        prediction_goal="Predict customer churn.",
        status="completed",
        current_stage="complete",
        trace_id="0123456789abcdef0123456789abcdef",
        artifact_references={"model": "mlflow://models/churn/7"},
        data_prep_result=data_prep(),
        model_selection_result=model_selection(),
        evaluation_result=evaluation(),
        deployment_result=deployment(),
    )


def test_all_six_models_publish_closed_json_schemas() -> None:
    models = (
        PipelineState,
        DataPrepResult,
        ModelSelectionResult,
        EvaluationResult,
        DeploymentResult,
        PipelineResult,
    )

    for model in models:
        schema = model.model_json_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        json.dumps(schema)


def test_contracts_round_trip_through_json() -> None:
    state = completed_state()

    restored = PipelineState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.data_prep_result.dataset_hash == HASH


def test_data_prep_rejects_bad_hash() -> None:
    payload = data_prep().model_dump()
    payload["dataset_hash"] = "not-a-sha"

    with pytest.raises(ValidationError, match="dataset_hash"):
        DataPrepResult.model_validate(payload)


def test_data_prep_rejects_target_leakage() -> None:
    payload = data_prep().model_dump()
    payload["feature_columns"] = ["age", "churned"]

    with pytest.raises(ValidationError, match="target_column must not be included"):
        DataPrepResult.model_validate(payload)


def test_model_selection_requires_best_run_to_be_a_candidate() -> None:
    payload = model_selection().model_dump()
    payload["best_run_id"] = "unknown-run"

    with pytest.raises(ValidationError, match="best_run_id"):
        ModelSelectionResult.model_validate(payload)


def test_evaluation_decision_must_match_threshold() -> None:
    payload = evaluation().model_dump()
    payload["passed"] = False
    payload["retry_feedback"] = "Try again."

    with pytest.raises(ValidationError, match="inconsistent"):
        EvaluationResult.model_validate(payload)


def test_failed_evaluation_requires_actionable_retry_feedback() -> None:
    payload = evaluation(passed=False).model_dump()
    payload["retry_feedback"] = None

    with pytest.raises(ValidationError, match="retry_feedback"):
        EvaluationResult.model_validate(payload)


def test_pipeline_state_rejects_exhausted_retry_budget() -> None:
    payload = completed_state().model_dump()
    payload.update(status="running", current_stage="evaluation", retry_count=3)

    with pytest.raises(ValidationError, match="retry_count cannot exceed"):
        PipelineState.model_validate(payload)


def test_completed_state_requires_all_successful_stage_results() -> None:
    payload = completed_state().model_dump()
    payload["deployment_result"] = None

    with pytest.raises(ValidationError, match="every stage result"):
        PipelineState.model_validate(payload)


def test_failed_state_requires_a_sanitized_error() -> None:
    with pytest.raises(ValidationError, match="sanitized error"):
        PipelineState(
            run_id=RUN_ID,
            dataset_uri="file:///datasets/churn.csv",
            task_type="classification",
            target_column="churned",
            prediction_goal="Predict customer churn.",
            status="failed",
            current_stage="data_preparation",
            trace_id="0123456789abcdef",
        )


def test_completed_pipeline_result_is_machine_serializable() -> None:
    result = PipelineResult(
        run_id=RUN_ID,
        status="completed",
        model_card_markdown="# Churn model\n\nAccepted on held-out data.",
        endpoint="https://churn.example.run.app",
        lineage={"dataset_sha256": HASH, "model_run_id": "run-forest"},
        trace_url="https://langfuse.example.com/trace/0123",
        evaluation=evaluation(),
        deployment=deployment(),
        runtime_seconds=42.5,
        input_tokens=1200,
        output_tokens=300,
        estimated_cost_usd=0.08,
    )

    payload = json.loads(result.model_dump_json())

    assert payload["status"] == "completed"
    assert payload["endpoint"].rstrip("/") == "https://churn.example.run.app"
    assert payload["lineage"]["dataset_sha256"] == HASH


def test_completed_pipeline_result_requires_matching_healthy_deployment() -> None:
    with pytest.raises(ValidationError, match="healthy deployment"):
        PipelineResult(
            run_id=RUN_ID,
            status="completed",
            model_card_markdown="# Failed smoke test",
            endpoint="https://churn.example.run.app",
            lineage={"dataset_sha256": HASH},
            trace_url="https://langfuse.example.com/trace/0123",
            evaluation=evaluation(),
            deployment=deployment(health_status="unhealthy"),
            runtime_seconds=42.5,
        )


def test_evaluation_completed_pipeline_result_does_not_claim_deployment() -> None:
    result = PipelineResult(
        run_id=RUN_ID,
        status="evaluation_completed",
        model_card_markdown="# Accepted model\n\nDeployment is outside this run.",
        lineage={"dataset_sha256": HASH, "model_run_id": "run-forest"},
        trace_url="http://localhost/traces/0123",
        evaluation=evaluation(),
        runtime_seconds=12.5,
    )

    assert result.evaluation is not None and result.evaluation.passed
    assert result.endpoint is None
    assert result.deployment is None


def test_extra_fields_fail_closed() -> None:
    payload = model_selection().model_dump()
    payload["raw_training_rows"] = [{"secret": "must not cross boundary"}]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelSelectionResult.model_validate(payload)
