"""Cache of validated LLM decisions, keyed by their exact row-free inputs.

Re-running the same dataset (same SHA-256 profile, target, goal, metrics and
retry feedback) with the same model asks the same question; the cached,
already-validated answer is reused instead of paying for another call.

* Only decisions that came from a live LLM and passed validation are stored.
* A cache hit is labelled ``source="cache"`` everywhere (spans, CLI, usage
  report), so nothing pretends to be a fresh LLM call.
* ``LLM_CACHE=off`` disables it (e.g. to demo fully live reasoning).
* Files live under ``LLM_CACHE_DIR`` (default ``artifacts/llm_cache``), one
  JSON file per decision, so the ADK subprocess and the orchestrator share it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def enabled() -> bool:
    return os.getenv("LLM_CACHE", "on").strip().casefold() not in {"off", "0", "false", "no"}


_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "llm_cache"


def _directory() -> Path:
    configured = os.getenv("LLM_CACHE_DIR", "").strip()
    return Path(configured) if configured else _DEFAULT_DIR


def key(stage: str, inputs: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "stage": stage,
            "model": os.getenv("CREWAI_MODEL", ""),
            "inputs": inputs,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def get(stage: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    if not enabled():
        return None
    path = _directory() / f"{key(stage, inputs)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def put(stage: str, inputs: dict[str, Any], decision: dict[str, Any]) -> None:
    if not enabled():
        return
    directory = _directory()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key(stage, inputs)}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(decision, default=str), encoding="utf-8")
        temporary.replace(target)
    except OSError:
        pass  # caching is an optimisation; never fail a run because of it
