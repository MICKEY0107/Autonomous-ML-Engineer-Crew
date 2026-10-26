"""Standalone prediction service for a deployed bundle (used on Cloud Run).

The bundle directory holds ``model.joblib`` and ``metadata.json`` written by
the Deployment Agent. ``/predict`` requires the ``X-API-Key`` header when
``PREDICT_API_KEY`` is set.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent))
MODEL = joblib.load(MODEL_DIR / "model.joblib")
METADATA = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
FEATURES = [item["name"] for item in METADATA["features"]]

app = FastAPI(title=f"Model {METADATA['run_id']}", version="1.0.0")


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1_000)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "run_id": METADATA["run_id"], "model": METADATA["model_name"]}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {"features": METADATA["features"], "target": METADATA["target_column"]}


@app.post("/predict")
def predict(request: PredictRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    expected = os.getenv("PREDICT_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
    frame = pd.DataFrame([{name: row.get(name) for name in FEATURES} for row in request.rows])
    for item in METADATA["features"]:
        if item["type"] == "number":
            frame[item["name"]] = pd.to_numeric(frame[item["name"]], errors="coerce")
    predictions = MODEL.predict(frame)
    return {"predictions": [value.item() if hasattr(value, "item") else value for value in predictions]}
