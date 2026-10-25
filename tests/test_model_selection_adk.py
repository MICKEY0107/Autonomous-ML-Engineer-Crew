"""Tests for deterministic ADK model selection and its A2A contract."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest
from starlette.testclient import TestClient

from agents.model_selection_adk import (
    A2A_HOST,
    A2A_PORT,
    ModelSelectionError,
    ModelSelectionRequest,
    app,
    parse_task_message,
    root_agent,
    select_models,
)
from contracts.models import DataPrepResult, ModelSelectionResult


CLASSIFICATION_RUN_ID = UUID("10000000-0000-4000-8000-000000000001")
REGRESSION_RUN_ID = UUID("10000000-0000-4000-8000-000000000002")


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory) -> str:
    database = tmp_path_factory.mktemp("model-selection-mlflow") / "mlflow.db"
    return f"sqlite:///{database.resolve().as_posix()}"


def _data_prep_result(
    tmp_path: Path,
    frame: pd.DataFrame,
    run_id: UUID,
) -> DataPrepResult:
    train_path = tmp_path / f"{run_id}-train.csv"
    test_path = tmp_path / f"{run_id}-test.csv"
    frame.to_csv(train_path, index=False)
    frame.iloc[:4].to_csv(test_path, index=False)
    return DataPrepResult(
        run_id=run_id,
        dataset_hash="a" * 64,
        schema_summary={column: str(dtype) for column, dtype in frame.dtypes.items()},
        quality_findings=["No data-quality issues were detected."],
        target_column="target",
        feature_columns=[column for column in frame.columns if column != "target"],
        row_count=len(frame) + 4,
        train_row_count=len(frame),
        test_row_count=4,
        train_artifact_uri=train_path.resolve().as_uri(),
        test_artifact_uri=test_path.resolve().as_uri(),
    )


def test_classification_trains_three_candidates_and_registers_best(
    tmp_path,
    tracking_uri,
) -> None:
    frame = pd.DataFrame(
        {
            "amount": [float(index) for index in range(40)],
            "region": ["north", "south", "east", "west"] * 10,
            "target": [index % 2 for index in range(40)],
        }
    )
    data_prep = _data_prep_result(tmp_path, frame, CLASSIFICATION_RUN_ID)
    request = ModelSelectionRequest(
        data_prep_result=data_prep,
        task_type="classification",
        tracking_uri=tracking_uri,
        experiment_name="classification-selection",
    )

    result = select_models(request, artifact_dir=tmp_path / "models")

    assert isinstance(result, ModelSelectionResult)
    assert result.run_id == CLASSIFICATION_RUN_ID
    assert len(result.candidate_run_ids) == 3
    assert result.best_run_id in result.candidate_run_ids
    assert result.best_parameters["model_type"] in {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert set(result.training_metrics) == {"accuracy", "f1_weighted"}
    assert result.best_model_uri.startswith("models:/autonomous-classification-")


def test_regression_trains_linear_rf_and_ridge_candidates(
    tmp_path,
    tracking_uri,
) -> None:
    frame = pd.DataFrame(
        {
            "x": [float(index) for index in range(40)],
            "segment": ["a", "b"] * 20,
            "target": [3.0 * index + (index % 2) for index in range(40)],
        }
    )
    data_prep = _data_prep_result(tmp_path, frame, REGRESSION_RUN_ID)
    request = ModelSelectionRequest(
        data_prep_result=data_prep,
        task_type="regression",
        tracking_uri=tracking_uri,
        experiment_name="regression-selection",
    )

    result = select_models(request, artifact_dir=tmp_path / "models")

    assert len(result.candidate_run_ids) == 3
    assert result.best_parameters["model_type"] in {
        "linear_regression",
        "random_forest",
        "ridge",
    }
    assert {"rmse", "mae"}.issubset(result.training_metrics)
    assert result.best_model_uri.startswith("models:/autonomous-regression-")


def test_a2a_service_contract_is_strict_and_bound_to_localhost(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "feature": list(range(8)),
            "target": [0, 1] * 4,
        }
    )
    data_prep = _data_prep_result(tmp_path, frame, CLASSIFICATION_RUN_ID)
    request = ModelSelectionRequest(
        data_prep_result=data_prep,
        task_type="classification",
    )

    parsed = parse_task_message(request.model_dump_json())

    assert parsed.data_prep_result == data_prep
    assert root_agent.name == "model_selection_agent"
    assert type(app).__name__ == "Starlette"
    assert (A2A_HOST, A2A_PORT) == ("127.0.0.1", 8080)
    with TestClient(app) as client:
        response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "model_selection_agent"
    assert card["supportedInterfaces"][0]["url"] == "http://127.0.0.1:8080"
    with pytest.raises(ModelSelectionError, match="valid ModelSelectionRequest JSON"):
        parse_task_message('{"data_prep_result": {}, "unexpected": true}')
