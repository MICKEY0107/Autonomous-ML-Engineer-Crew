"""Tests for the MLflow experiment MCP tool."""

from __future__ import annotations

import tempfile

from mlflow import MlflowClient
import pytest

from tools.log_experiment import ExperimentResult, log_experiment


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory) -> str:
    database_path = tmp_path_factory.mktemp("mlflow-store") / "mlflow.db"
    return f"sqlite:///{database_path.resolve().as_posix()}"


def test_create_run_and_log_params_and_metrics(tracking_uri) -> None:
    created = log_experiment(
        "create_run",
        tracking_uri,
        experiment_name="classification",
        run_name="random-forest",
    )
    with_params = log_experiment(
        "log_params",
        tracking_uri,
        run_id=created.run_id,
        params={"n_estimators": 100, "class_weight": "balanced"},
    )
    with_metrics = log_experiment(
        "log_metrics",
        tracking_uri,
        run_id=created.run_id,
        metrics={"f1": 0.86, "roc_auc": 0.91},
        finish_run=True,
    )

    assert isinstance(created, ExperimentResult)
    assert created.status == "RUNNING"
    assert with_params.params == {
        "n_estimators": "100",
        "class_weight": "balanced",
    }
    assert with_metrics.metrics == {"f1": 0.86, "roc_auc": 0.91}
    assert with_metrics.status == "FINISHED"


def test_query_best_run_respects_metric_direction(tracking_uri) -> None:
    run_ids: list[str] = []
    for run_name, rmse in (("baseline", 1.2), ("tuned", 0.7)):
        created = log_experiment(
            "create_run",
            tracking_uri,
            experiment_name="regression",
            run_name=run_name,
        )
        run_ids.append(created.run_id)
        log_experiment(
            "log_metrics",
            tracking_uri,
            run_id=created.run_id,
            metrics={"rmse": rmse},
            finish_run=True,
        )

    best = log_experiment(
        "query_best_run",
        tracking_uri,
        experiment_name="regression",
        metric_name="rmse",
        maximize=False,
    )

    assert best.run_id == run_ids[1]
    assert best.metrics["rmse"] == 0.7
    assert best.status == "FINISHED"


def test_register_model_creates_version_from_run_artifact(
    tracking_uri, tmp_path, monkeypatch
) -> None:
    created = log_experiment(
        "create_run",
        tracking_uri,
        experiment_name="registry",
        run_name="candidate",
    )
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client.log_text(created.run_id, "model metadata", "model/MLmodel")

    registered = log_experiment(
        "register_model",
        tracking_uri,
        run_id=created.run_id,
        artifact_path="model",
        registered_model_name="customer-churn",
    )

    assert registered.registered_model_name == "customer-churn"
    assert registered.model_version == "1"
    assert registered.model_uri == f"runs:/{created.run_id}/model"
    assert client.get_model_version("customer-churn", "1").run_id == created.run_id
