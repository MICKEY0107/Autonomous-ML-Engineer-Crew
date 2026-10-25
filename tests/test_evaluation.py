"""Integration tests for the CrewAI held-out evaluation stage."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from agents.evaluation import (
    EvaluateModelTool,
    build_evaluation_agent,
    evaluate_model,
)
from agents.model_selection_adk import ModelSelectionRequest, select_models
from contracts.models import DataPrepResult, EvaluationResult


def _data_prep_result(
    root: Path,
    run_id: UUID,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> DataPrepResult:
    train_path = root / f"{run_id}-train.csv"
    test_path = root / f"{run_id}-test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    combined = pd.concat([train, test], ignore_index=True)
    return DataPrepResult(
        run_id=run_id,
        dataset_hash="b" * 64,
        schema_summary={column: str(dtype) for column, dtype in combined.dtypes.items()},
        quality_findings=["No data-quality issues were detected."],
        target_column="target",
        feature_columns=[column for column in combined.columns if column != "target"],
        row_count=len(combined),
        train_row_count=len(train),
        test_row_count=len(test),
        train_artifact_uri=train_path.resolve().as_uri(),
        test_artifact_uri=test_path.resolve().as_uri(),
    )


@pytest.fixture(scope="module")
def evaluation_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation-cases")
    tracking_uri = f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}"

    classification_train = pd.DataFrame(
        {
            "score": [float(index) for index in range(40)],
            "channel": ["web", "store"] * 20,
            "target": [0 if index < 20 else 1 for index in range(40)],
        }
    )
    classification_test = pd.DataFrame(
        {
            "score": [2.0, 7.0, 14.0, 19.0, 21.0, 27.0, 34.0, 39.0],
            "channel": ["web", "store"] * 4,
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    classification_data = _data_prep_result(
        root,
        UUID("20000000-0000-4000-8000-000000000001"),
        classification_train,
        classification_test,
    )
    classification_selection = select_models(
        ModelSelectionRequest(
            data_prep_result=classification_data,
            task_type="classification",
            tracking_uri=tracking_uri,
            experiment_name="evaluation-classification",
        ),
        artifact_dir=root / "models",
    )

    regression_train = pd.DataFrame(
        {
            "x": [float(index) for index in range(40)],
            "group": ["a", "b"] * 20,
            "target": [2.0 * index for index in range(40)],
        }
    )
    regression_test = pd.DataFrame(
        {
            "x": [41.0, 42.0, 43.0, 44.0, 45.0, 46.0],
            "group": ["a", "b"] * 3,
            "target": [132.0, 134.0, 136.0, 138.0, 140.0, 142.0],
        }
    )
    regression_data = _data_prep_result(
        root,
        UUID("20000000-0000-4000-8000-000000000002"),
        regression_train,
        regression_test,
    )
    regression_selection = select_models(
        ModelSelectionRequest(
            data_prep_result=regression_data,
            task_type="regression",
            tracking_uri=tracking_uri,
            experiment_name="evaluation-regression",
        ),
        artifact_dir=root / "models",
    )
    return root, tracking_uri, classification_selection, regression_selection


def test_classification_returns_passing_held_out_metrics(evaluation_cases) -> None:
    root, tracking_uri, selection, _ = evaluation_cases

    result = evaluate_model(
        selection,
        tracking_uri=tracking_uri,
        acceptance_threshold=0.80,
        artifact_dir=root / "downloads",
    )

    assert isinstance(result, EvaluationResult)
    assert result.run_id == selection.run_id
    assert result.model_uri == selection.best_model_uri
    assert result.primary_metric == "f1_weighted"
    assert {"accuracy", "f1_weighted", "roc_auc"}.issubset(
        result.held_out_metrics
    )
    assert result.passed is True
    assert result.retry_feedback is None


def test_regression_failure_returns_actionable_feedback(evaluation_cases) -> None:
    root, tracking_uri, _, selection = evaluation_cases

    result = evaluate_model(
        selection,
        tracking_uri=tracking_uri,
        acceptance_threshold=0.0,
        artifact_dir=root / "downloads",
    )

    assert result.primary_metric == "rmse"
    assert {"rmse", "mae", "r2"}.issubset(result.held_out_metrics)
    assert result.held_out_metrics["rmse"] > 0
    assert result.passed is False
    assert result.retry_feedback is not None
    assert "residuals" in result.retry_feedback


def test_crewai_tool_returns_evaluation_result_json(evaluation_cases) -> None:
    root, tracking_uri, selection, _ = evaluation_cases
    tool = EvaluateModelTool(
        default_tracking_uri=tracking_uri,
        artifact_dir=str(root / "tool-downloads"),
    )

    payload = tool._run(
        model_selection_result=selection,
        acceptance_threshold=0.80,
    )
    result = EvaluationResult.model_validate_json(payload)

    assert result.passed is True
    agent = build_evaluation_agent(
        tracking_uri=tracking_uri,
        artifact_dir=root / "tool-downloads",
    )
    assert agent.role == "Model Evaluation Engineer"
    assert [registered_tool.name for registered_tool in agent.tools] == [
        "evaluate_model"
    ]
