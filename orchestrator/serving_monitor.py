"""Runtime monitoring and request limits for deployed models.

Every call to a deployment's ``/predict`` or ``/predict_csv`` is recorded:
request and row counts, errors, latency (p50/p95 over the last 500 calls) and
the last error. The Deployment and Observability Agent's serving policy sets
the limits (rows per request, requests per minute) and the alert thresholds
(p95 latency, error rate); this module enforces and evaluates them.

Counters are kept in memory and mirrored to ``monitoring.json`` in the
deployment's bundle folder, so they survive a server restart.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_POLICY = {
    "max_rows_per_request": 500,
    "rate_limit_per_minute": 120,
    "latency_alert_ms": 500,
    "error_rate_alert_pct": 5.0,
    "reasoning": "Default serving policy (bundle created before serving policies existed).",
    "source": "fallback",
}
_LATENCY_WINDOW = 500
_MIN_REQUESTS_FOR_ALERTS = 5

_lock = threading.Lock()
_stats: dict[str, dict[str, Any]] = {}


def policy_of(metadata: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_POLICY, **(metadata.get("serving_policy") or {})}


def _fresh() -> dict[str, Any]:
    return {
        "requests": 0,
        "rows": 0,
        "errors": 0,
        "rejected": 0,
        "latencies_ms": deque(maxlen=_LATENCY_WINDOW),
        "recent_calls": deque(),
        "last_error": None,
        "last_request_at": None,
        "since": datetime.now(timezone.utc).isoformat(),
    }


def _load(run_hex: str, folder: Path | None) -> dict[str, Any]:
    stats = _stats.get(run_hex)
    if stats is not None:
        return stats
    stats = _fresh()
    saved = folder / "monitoring.json" if folder is not None else None
    if saved is not None and saved.is_file():
        try:
            data = json.loads(saved.read_text(encoding="utf-8"))
            for key in ("requests", "rows", "errors", "rejected", "last_error", "last_request_at", "since"):
                if key in data:
                    stats[key] = data[key]
            stats["latencies_ms"].extend(float(v) for v in data.get("latencies_ms", []))
        except (OSError, ValueError, TypeError):
            pass
    _stats[run_hex] = stats
    return stats


def _save(stats: dict[str, Any], folder: Path | None) -> None:
    if folder is None or not folder.is_dir():
        return
    data = {key: value for key, value in stats.items() if key not in {"latencies_ms", "recent_calls"}}
    data["latencies_ms"] = list(stats["latencies_ms"])
    try:
        (folder / "monitoring.json").write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def check_limits(run_hex: str, policy: dict[str, Any], rows: int, folder: Path | None = None) -> str | None:
    """Return a refusal reason (and count it) if the request breaks the policy."""

    now = time.monotonic()
    with _lock:
        stats = _load(run_hex, folder)
        window = stats["recent_calls"]
        while window and now - window[0] > 60:
            window.popleft()
        reason = None
        if rows > int(policy["max_rows_per_request"]):
            reason = (
                f"too many rows: {rows} > {policy['max_rows_per_request']} per request "
                "(serving policy)"
            )
        elif len(window) >= int(policy["rate_limit_per_minute"]):
            reason = (
                f"rate limit reached: {policy['rate_limit_per_minute']} requests per "
                "minute (serving policy)"
            )
        if reason:
            stats["rejected"] += 1
            _save(stats, folder)
            return reason
        window.append(now)
        return None


def record(
    run_hex: str,
    *,
    rows: int,
    latency_ms: float,
    error: str | None = None,
    folder: Path | None = None,
) -> None:
    with _lock:
        stats = _load(run_hex, folder)
        stats["requests"] += 1
        stats["rows"] += rows if error is None else 0
        stats["latencies_ms"].append(round(float(latency_ms), 2))
        stats["last_request_at"] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            stats["errors"] += 1
            stats["last_error"] = error[:300]
        _save(stats, folder)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 2)


def snapshot(run_hex: str, policy: dict[str, Any], folder: Path | None = None) -> dict[str, Any]:
    with _lock:
        stats = _load(run_hex, folder)
        latencies = list(stats["latencies_ms"])
        requests = int(stats["requests"])
        errors = int(stats["errors"])
        data = {
            "requests": requests,
            "rows_scored": int(stats["rows"]),
            "errors": errors,
            "rejected_by_policy": int(stats["rejected"]),
            "error_rate_pct": round(100 * errors / requests, 2) if requests else 0.0,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            "last_error": stats["last_error"],
            "last_request_at": stats["last_request_at"],
            "since": stats["since"],
        }
    alerts = []
    if requests >= _MIN_REQUESTS_FOR_ALERTS:
        if data["latency_p95_ms"] is not None and data["latency_p95_ms"] > policy["latency_alert_ms"]:
            alerts.append(
                f"p95 latency {data['latency_p95_ms']:.0f} ms is above the {policy['latency_alert_ms']} ms alert"
            )
        if data["error_rate_pct"] > float(policy["error_rate_alert_pct"]):
            alerts.append(
                f"error rate {data['error_rate_pct']:.1f}% is above the {policy['error_rate_alert_pct']}% alert"
            )
    return {**data, "alerts": alerts, "status": "alert" if alerts else "ok", "policy": policy}
