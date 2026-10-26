"""Server-side run history (SQLite) so results survive refreshes and devices.

Stores one row per finished run: headline numbers for the history table and
trend charts, plus the full dashboard payload (result, insights, decisions,
A2A log) as JSON for reopening a past run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
DEFAULT_DB = Path(__file__).resolve().parents[1] / "artifacts" / "ui" / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset_name TEXT,
    target_column TEXT,
    prediction_goal TEXT,
    task_type TEXT,
    model_name TEXT,
    metric_name TEXT,
    metric_value REAL,
    runtime_seconds REAL,
    cost_usd REAL,
    tokens INTEGER,
    llm_calls INTEGER,
    cache_hits INTEGER,
    retries INTEGER,
    payload TEXT NOT NULL
)
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(_SCHEMA)
    return connection


def save_run(payload: dict[str, Any], path: Path = DEFAULT_DB) -> None:
    final = payload.get("final_result") or {}
    usage = final.get("usage_report") or {}
    meta = payload.get("run_input") or {}
    row = (
        final.get("run_id"),
        final.get("completed_at"),
        final.get("status"),
        meta.get("dataset_name"),
        meta.get("target_column"),
        meta.get("prediction_goal"),
        (payload.get("insights") or {}).get("dataset", {}).get("task_type"),
        final.get("model_name"),
        final.get("metric_name"),
        final.get("metric_value"),
        (final.get("runtime_ms") or 0) / 1000,
        final.get("estimated_cost_usd") or 0.0,
        final.get("token_count") or 0,
        usage.get("llm_requests") or 0,
        usage.get("cache_hits") or 0,
        final.get("retry_count") or 0,
        json.dumps(payload, default=str),
    )
    with _LOCK, _connect(path) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row
        )


def list_runs(limit: int = 50, path: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    with _LOCK, _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{key: row[key] for key in row.keys() if key != "payload"} for row in rows]


def get_run(run_id: str, path: Path = DEFAULT_DB) -> dict[str, Any] | None:
    with _LOCK, _connect(path) as connection:
        row = connection.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return json.loads(row["payload"]) if row else None
