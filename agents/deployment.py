"""Deployment and Observability Agent: package the accepted model and serve it.

Runs only after the held-out evaluation passes. Deployment is deliberately
deterministic (no LLM): packaging, hashing and health checks must be exact.

Targets (``DEPLOY_TARGET``):

* ``local`` (default) - the bundle is served by the dashboard API at
  ``/api/deployments/<run>/predict``; the health check loads the bundle and
  scores one held-out row.
* ``cloudrun`` - the bundle plus ``serving/`` (FastAPI + Dockerfile) is
  deployed with ``gcloud run deploy --source``; the health check calls the
  live service's ``/health``. Needs the Google Cloud CLI logged in, and
  ``GCP_PROJECT`` / ``GCP_REGION``.

Either way the result is a validated ``DeploymentResult`` with a SHA-256
digest of what was deployed, the service URL, revision and health.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from contracts.models import DeploymentResult, HealthStatus, PipelineState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_ROOT = PROJECT_ROOT / "artifacts" / "deployments"
SERVING_SOURCE = PROJECT_ROOT / "serving"


class DeploymentError(RuntimeError):
    """Packaging or serving the model failed."""


def bundle_dir(run_hex: str) -> Path:
    return DEPLOYMENT_ROOT / run_hex


def _feature_schema(train_csv: Path, features: list[str]) -> list[dict[str, Any]]:
    frame = pd.read_csv(train_csv, usecols=features)
    schema = []
    for column in features:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            schema.append(
                {
                    "name": column,
                    "type": "number",
                    "example": float(clean.median()) if not clean.empty else 0.0,
                    "min": float(clean.min()) if not clean.empty else None,
                    "max": float(clean.max()) if not clean.empty else None,
                }
            )
        else:
            # Blank cells are missing values, not a category called "nan".
            counts = series.dropna().astype(str).value_counts()
            schema.append(
                {
                    "name": column,
                    "type": "category",
                    "example": str(counts.index[0]) if not counts.empty else "",
                    "categories": [str(value) for value in counts.index[:30]],
                }
            )
    return schema


def _digest(paths: list[Path]) -> str:
    sha = hashlib.sha256()
    for path in paths:
        sha.update(path.read_bytes())
    return f"sha256:{sha.hexdigest()}"


def load_bundle(run_hex: str) -> tuple[Any, dict[str, Any]]:
    directory = bundle_dir(run_hex)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(directory / "model.joblib")
    return model, metadata


def predict_rows(model: Any, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score rows (missing features are left empty and imputed by the pipeline)."""

    features = [item["name"] for item in metadata["features"]]
    frame = pd.DataFrame([{name: row.get(name) for name in features} for row in rows])
    for item in metadata["features"]:
        column = item["name"]
        if item["type"] == "number":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            # Missing categories -> NaN so the pipeline's imputer recognises them.
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
            frame[column] = frame[column].map(lambda v: v if v is np.nan else str(v))
    predictions = model.predict(frame)
    output = [{"prediction": _jsonable(value)} for value in predictions]
    predict_proba = getattr(model, "predict_proba", None)
    if metadata["task_type"] == "classification" and callable(predict_proba):
        try:
            probabilities = predict_proba(frame)
            classes = [str(value) for value in getattr(model, "classes_", [])]
            for item, row in zip(output, probabilities):
                item["probabilities"] = {cls: round(float(p), 4) for cls, p in zip(classes, row)}
        except (TypeError, ValueError, AttributeError):
            pass
    return output


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    return round(value, 6) if isinstance(value, float) else value


def _package(state: PipelineState, tracking_uri: str) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    from mlflow import MlflowClient

    from agents.evaluation import _download_evaluation_bundle

    selection = state.model_selection_result
    evaluation = state.evaluation_result
    if selection is None or evaluation is None or not evaluation.passed:
        raise DeploymentError("only a model that passed held-out evaluation is deployed")
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    run_hex = state.run_id.hex
    directory = bundle_dir(run_hex)
    model, held_out, eval_meta = _download_evaluation_bundle(
        client, selection, directory / "download-cache"
    )
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    from orchestrator.flow import _local_path  # local import: flow imports this module

    train_csv = _local_path(state.data_prep_result.train_artifact_uri)
    metadata = {
        "run_id": str(state.run_id),
        "task_type": str(state.task_type),
        "target_column": eval_meta.target_column,
        "prediction_goal": state.prediction_goal,
        "model_name": selection.best_model_name,
        "model_uri": selection.best_model_uri,
        "held_out_metrics": evaluation.held_out_metrics,
        "primary_metric": evaluation.primary_metric,
        "features": _feature_schema(train_csv, eval_meta.feature_columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.rmtree(directory / "download-cache", ignore_errors=True)
    return directory, metadata, held_out[eval_meta.feature_columns]


def _deploy_local(state: PipelineState, directory: Path, metadata: dict[str, Any], sample: pd.DataFrame) -> DeploymentResult:
    run_hex = state.run_id.hex
    base_url = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        model, loaded = load_bundle(run_hex)
        predict_rows(model, loaded, sample.head(1).to_dict(orient="records"))
        health = HealthStatus.HEALTHY
    except Exception:
        health = HealthStatus.UNHEALTHY
    return DeploymentResult(
        run_id=state.run_id,
        image_digest=_digest([directory / "model.joblib", directory / "metadata.json"]),
        service_url=f"{base_url}/api/deployments/{run_hex}",
        health_status=health,
        deployment_revision=f"local-{run_hex[:8]}-r1",
        deployed_at=datetime.now(timezone.utc),
    )


def _deploy_cloudrun(state: PipelineState, directory: Path) -> DeploymentResult:
    project = os.environ.get("GCP_PROJECT")
    region = os.getenv("GCP_REGION", "asia-south1")
    if not project:
        raise DeploymentError("GCP_PROJECT is required for DEPLOY_TARGET=cloudrun")
    gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if gcloud is None:
        raise DeploymentError("Google Cloud CLI (gcloud) is not installed")
    for name in ("app.py", "Dockerfile", "requirements.txt"):
        shutil.copy2(SERVING_SOURCE / name, directory / name)
    service = f"amec-{state.run_id.hex[:12]}"
    env_vars = [f"MODEL_DIR=/app"]
    if os.getenv("PREDICT_API_KEY"):
        env_vars.append(f"PREDICT_API_KEY={os.environ['PREDICT_API_KEY']}")
    subprocess.run(
        [
            gcloud, "run", "deploy", service, "--source", str(directory),
            "--project", project, "--region", region, "--allow-unauthenticated",
            "--memory", "1Gi", "--max-instances", "2", "--quiet",
            "--set-env-vars", ",".join(env_vars),
        ],
        check=True, capture_output=True, text=True, timeout=1_200,
    )
    described = json.loads(
        subprocess.run(
            [gcloud, "run", "services", "describe", service, "--project", project,
             "--region", region, "--format", "json"],
            check=True, capture_output=True, text=True, timeout=120,
        ).stdout
    )
    url = described["status"]["url"]
    revision = described["status"].get("latestReadyRevisionName", service)
    image = described["spec"]["template"]["spec"]["containers"][0].get("image", "")
    digest = image.split("@", 1)[1] if "@sha256:" in image else _digest([directory / "model.joblib"])
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=30) as response:  # noqa: S310
            health = HealthStatus.HEALTHY if response.status == 200 else HealthStatus.UNHEALTHY
    except Exception:
        health = HealthStatus.UNHEALTHY
    return DeploymentResult(
        run_id=state.run_id,
        image_digest=digest,
        service_url=url,
        health_status=health,
        deployment_revision=revision,
        deployed_at=datetime.now(timezone.utc),
    )


def deploy_model(state: PipelineState, *, tracking_uri: str) -> DeploymentResult:
    directory, metadata, sample = _package(state, tracking_uri)
    target = os.getenv("DEPLOY_TARGET", "local").strip().casefold()
    if target == "cloudrun":
        return _deploy_cloudrun(state, directory)
    return _deploy_local(state, directory, metadata, sample)
