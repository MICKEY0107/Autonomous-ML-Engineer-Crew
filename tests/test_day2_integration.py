"""Day 2 integration checks for contracts, tools, and all three agents."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from agents.data_prep import run_data_prep
from agents.evaluation import run_evaluation
from agents.model_selection_adk import ModelSelectionRequest
from orchestrator.a2a_client import send_model_selection_task
from contracts.models import (
    DataPrepResult,
    DeploymentResult,
    EvaluationResult,
    ModelSelectionResult,
    PipelineResult,
    PipelineState,
)
from tools.log_experiment import log_and_query_experiment, query_best_run
from tools.profile_dataset import ProfileResult, profile_dataset


RUN_ID = UUID("30000000-0000-4000-8000-000000000001")
DATASET_HASH = "c" * 64


def _dummy_data_prep() -> DataPrepResult:
    return DataPrepResult(
        run_id=RUN_ID,
        dataset_hash=DATASET_HASH,
        schema_summary={"feature1": "float64", "target": "int64"},
        quality_findings=["No data-quality issues were detected."],
        target_column="target",
        feature_columns=["feature1"],
        row_count=50,
        train_row_count=40,
        test_row_count=10,
        train_artifact_uri="file:///tmp/day2-train.csv",
        test_artifact_uri="file:///tmp/day2-test.csv",
    )


def _mock_selection(metric_value: float) -> ModelSelectionResult:
    return ModelSelectionResult(
        run_id=RUN_ID,
        candidate_run_ids=["run-logreg", "run-rf", "run-gbm"],
        best_run_id="run-rf",
        best_model_uri="models:/day2-model/1",
        best_parameters={"model_type": "random_forest", "n_estimators": 100},
        training_metrics={"accuracy": metric_value},
    )


def test_all_contracts_instantiate() -> None:
    data_prep = _dummy_data_prep()
    selection = _mock_selection(0.85)
    evaluation = EvaluationResult(
        run_id=RUN_ID,
        model_uri=selection.best_model_uri,
        held_out_metrics={"accuracy": 0.85},
        primary_metric="accuracy",
        acceptance_threshold=0.80,
        metric_direction="maximize",
        threshold_rationale="Accuracy must meet the Day 2 acceptance gate.",
        passed=True,
    )
    deployment = DeploymentResult(
        run_id=RUN_ID,
        image_digest=f"sha256:{'d' * 64}",
        service_url="https://day2.example.run.app",
        health_status="healthy",
        deployment_revision="day2-00001",
        deployed_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
    )
    state = PipelineState(
        run_id=RUN_ID,
        dataset_uri="file:///tmp/test_input.csv",
        task_type="classification",
        target_column="target",
        prediction_goal="Predict the synthetic target.",
        status="running",
        current_stage="evaluation",
        trace_id="0123456789abcdef",
        data_prep_result=data_prep,
        model_selection_result=selection,
        evaluation_result=evaluation,
    )
    pipeline_result = PipelineResult(
        run_id=RUN_ID,
        status="failed",
        model_card_markdown="# Day 2 smoke-test result",
        lineage={"dataset_sha256": DATASET_HASH},
        trace_url="https://langfuse.example/trace/day2",
        runtime_seconds=1.0,
        error="Deployment was intentionally skipped in this contract test.",
    )

    instances = (
        state,
        data_prep,
        selection,
        evaluation,
        deployment,
        pipeline_result,
    )
    assert all(isinstance(instance, BaseModel) for instance in instances)
    assert {"run_id", "dataset_uri", "task_type"}.issubset(
        PipelineState.model_fields
    )
    assert {"train_artifact_uri", "test_artifact_uri"}.issubset(
        DataPrepResult.model_fields
    )
    assert {"best_run_id", "training_metrics"}.issubset(
        ModelSelectionResult.model_fields
    )
    assert {"passed", "held_out_metrics", "retry_feedback"}.issubset(
        EvaluationResult.model_fields
    )
    assert {"service_url", "health_status"}.issubset(
        DeploymentResult.model_fields
    )
    assert {"status", "lineage", "trace_url"}.issubset(
        PipelineResult.model_fields
    )

    with pytest.raises(ValidationError):
        DataPrepResult.model_validate(
            data_prep.model_dump(exclude={"target_column"})
        )
    with pytest.raises(ValidationError):
        DeploymentResult.model_validate(
            deployment.model_dump(exclude={"service_url"})
        )


def test_profile_dataset_tool(small_csv: Path) -> None:
    result = profile_dataset("/tmp/test_input.csv")

    assert isinstance(result, ProfileResult)
    assert not isinstance(result, dict)
    assert isinstance(result.schema, dict)
    assert isinstance(result.null_counts, dict)
    assert isinstance(result.sha256, str)
    assert result.sha256
    assert set(result.schema) == {"feature1", "feature2", "feature3", "target"}


def test_log_experiment_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'day2-mlflow.db').resolve().as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    first = log_and_query_experiment(
        experiment_name="pytest_run",
        params={"model": "rf"},
        metrics={"accuracy": 0.85},
    )
    second = log_and_query_experiment(
        experiment_name="pytest_run",
        metrics={"accuracy": 0.91},
    )
    best = query_best_run("pytest_run", metric="accuracy")

    assert isinstance(first.run_id, str) and first.run_id
    assert isinstance(second.run_id, str) and second.run_id
    assert best.run_id == second.run_id
    assert best.metrics["accuracy"] == pytest.approx(0.91)


def test_data_prep_agent() -> None:
    generator = random.Random(7)
    csv_path = Path("/tmp/test_data.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "feature1": [generator.random() for _ in range(100)],
            "feature2": [generator.uniform(-5, 5) for _ in range(100)],
            "feature3": [generator.uniform(10, 20) for _ in range(100)],
            "target": [index % 2 for index in range(100)],
        }
    )
    frame.to_csv(csv_path, index=False)
    try:
        result = run_data_prep(
            csv_path="/tmp/test_data.csv",
            target_column="target",
            run_id="pytest-001",
        )
    finally:
        csv_path.unlink(missing_ok=True)

    assert isinstance(result, DataPrepResult)
    assert Path(result.train_path).exists()
    assert Path(result.test_path).exists()
    assert len(pd.read_csv(result.train_path)) > len(pd.read_csv(result.test_path))


def test_adk_agent_a2a() -> None:
    generator = random.Random(11)
    csv_path = Path("/tmp/test_a2a.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "feature1": [generator.random() for _ in range(80)],
            "feature2": [generator.uniform(-3, 3) for _ in range(80)],
            "feature3": [generator.uniform(5, 15) for _ in range(80)],
            "target": [index % 2 for index in range(80)],
        }
    )
    frame.to_csv(csv_path, index=False)
    prepared = run_data_prep(
        csv_path=csv_path,
        target_column="target",
        run_id=f"pytest-a2a-{uuid4()}",
    )
    tracking_uri = (
        f"sqlite:///{(Path('/tmp') / f'day2-a2a-{uuid4()}.db').resolve().as_posix()}"
    )
    request = ModelSelectionRequest(
        data_prep_result=prepared,
        task_type="classification",
        tracking_uri=tracking_uri,
        experiment_name="pytest-a2a",
    )
    environment = os.environ.copy()
    environment["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    project_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "agents.model_selection_adk"],
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)
        deadline = time.monotonic() + 45
        while True:
            if process.poll() is not None:
                pytest.fail("ADK A2A server exited before accepting requests")
            try:
                card = httpx.get(
                    "http://localhost:8080/.well-known/agent-card.json",
                    timeout=1.0,
                )
                if card.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                pytest.fail("ADK A2A server did not start within 48 seconds")
            time.sleep(0.5)

        # Real A2A protocol: Agent Card discovery + SendMessage JSON-RPC.
        result, call_info = send_model_selection_task(
            "http://localhost:8080",
            request.model_dump(mode="json"),
            timeout_seconds=120.0,
        )
        assert isinstance(result, ModelSelectionResult)
        assert call_info.agent_name == "model_selection_agent"
        assert call_info.task_state == "TASK_STATE_COMPLETED"

        assert isinstance(result.best_model_name, str)
        assert result.best_model_name
        assert isinstance(result.metric_value, float)
        assert 0.0 <= result.metric_value <= 1.0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        csv_path.unlink(missing_ok=True)


def test_evaluation_agent(small_csv: Path) -> None:
    failing_selection = _mock_selection(metric_value=0.45)
    failed = run_evaluation(failing_selection, small_csv)

    assert failed.passed is False
    assert isinstance(failed.feedback, str)
    assert failed.feedback

    passing_selection = _mock_selection(metric_value=0.85)
    passed = run_evaluation(passing_selection, small_csv)

    assert passed.passed is True
