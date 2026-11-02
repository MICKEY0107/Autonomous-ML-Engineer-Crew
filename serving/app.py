"""Standalone prediction service for a deployed bundle (used on Cloud Run).

The bundle directory holds ``model.joblib`` and ``metadata.json`` written by
the Deployment Agent. ``/predict`` requires the ``X-API-Key`` header when
``PREDICT_API_KEY`` is set.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent))
MODEL = joblib.load(MODEL_DIR / "model.joblib")
METADATA = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
FEATURES = [item["name"] for item in METADATA["features"]]

POLICY = {
    "max_rows_per_request": 500,
    "rate_limit_per_minute": 120,
    "latency_alert_ms": 500,
    "error_rate_alert_pct": 5.0,
    **(METADATA.get("serving_policy") or {}),
}
_LOCK = threading.Lock()
_STATS: dict[str, Any] = {"requests": 0, "rows": 0, "errors": 0, "rejected": 0, "last_error": None}
_LATENCIES: deque[float] = deque(maxlen=500)
_RECENT: deque[float] = deque()

app = FastAPI(title=f"Model {METADATA['run_id']}", version="1.0.0")


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1_000)


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Same conversion as agents.deployment.predict_rows (kept in sync)."""

    frame = pd.DataFrame([{name: row.get(name) for name in FEATURES} for row in rows])
    for item in METADATA["features"]:
        column = item["name"]
        if item["type"] == "number":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            # Missing categories -> NaN so the pipeline's imputer recognises them.
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
            frame[column] = frame[column].map(lambda v: v if v is np.nan else str(v))
    return frame


def _predict(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = _frame(rows)
    output = [
        {"prediction": value.item() if hasattr(value, "item") else value}
        for value in MODEL.predict(frame)
    ]
    predict_proba = getattr(MODEL, "predict_proba", None)
    if METADATA["task_type"] == "classification" and callable(predict_proba):
        try:
            classes = [str(value) for value in getattr(MODEL, "classes_", [])]
            for item, row in zip(output, predict_proba(frame)):
                item["probabilities"] = {cls: round(float(p), 4) for cls, p in zip(classes, row)}
        except (TypeError, ValueError, AttributeError):
            pass
    return output


@app.get("/health")
def health() -> dict[str, Any]:
    # Healthy only if the model can actually score a typical row.
    example = {item["name"]: item.get("example") for item in METADATA["features"]}
    try:
        _predict([example])
    except Exception as exc:  # noqa: BLE001 - health reports any scoring failure
        raise HTTPException(status_code=503, detail=f"model cannot score: {type(exc).__name__}")
    return {"status": "ok", "run_id": METADATA["run_id"], "model": METADATA["model_name"]}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {"features": METADATA["features"], "target": METADATA["target_column"]}


@app.post("/predict")
def predict(request: PredictRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    expected = os.getenv("PREDICT_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8"))):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
    now = time.monotonic()
    with _LOCK:
        while _RECENT and now - _RECENT[0] > 60:
            _RECENT.popleft()
        if len(request.rows) > int(POLICY["max_rows_per_request"]):
            _STATS["rejected"] += 1
            raise HTTPException(status_code=413, detail="too many rows for the serving policy")
        if len(_RECENT) >= int(POLICY["rate_limit_per_minute"]):
            _STATS["rejected"] += 1
            raise HTTPException(status_code=429, detail="rate limit reached (serving policy)")
        _RECENT.append(now)
    started = time.perf_counter()
    try:
        results = _predict(request.rows)
    except Exception as exc:
        with _LOCK:
            _STATS["requests"] += 1
            _STATS["errors"] += 1
            _STATS["last_error"] = type(exc).__name__
            _LATENCIES.append((time.perf_counter() - started) * 1000)
        raise HTTPException(status_code=422, detail=f"prediction failed: {type(exc).__name__}") from exc
    with _LOCK:
        _STATS["requests"] += 1
        _STATS["rows"] += len(request.rows)
        _LATENCIES.append((time.perf_counter() - started) * 1000)
    return {"predictions": [item["prediction"] for item in results], "results": results}


@app.get("/metrics")
def metrics(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Usage and error monitoring for this container, checked against the policy."""

    expected = os.getenv("PREDICT_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8"))):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
    with _LOCK:
        latencies = sorted(_LATENCIES)
        stats = dict(_STATS)
    p95 = round(latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))], 2) if latencies else None
    error_rate = round(100 * stats["errors"] / stats["requests"], 2) if stats["requests"] else 0.0
    alerts = []
    if stats["requests"] >= 5 and p95 is not None and p95 > POLICY["latency_alert_ms"]:
        alerts.append("p95 latency above the policy alert")
    if stats["requests"] >= 5 and error_rate > float(POLICY["error_rate_alert_pct"]):
        alerts.append("error rate above the policy alert")
    return {**stats, "error_rate_pct": error_rate, "latency_p95_ms": p95, "alerts": alerts, "policy": POLICY}
