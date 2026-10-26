"""Health of each configured LLM provider, without spending generation quota.

* Gemini     - fetch the configured model's metadata (no tokens generated).
* Ollama     - list local models and check the configured one is pulled.
* OpenRouter - read the key's remaining credit from ``/api/v1/key``.

Results are cached for 60 seconds so the dashboard can poll cheaply.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from orchestrator import decision_cache
from orchestrator.llm_provider import provider_of

_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
TTL_SECONDS = 60


def _models_in_order() -> list[dict[str, str]]:
    primary = os.getenv("CREWAI_MODEL", "")
    chain = [m.strip() for m in os.getenv("LLM_FALLBACK_CHAIN", "").split(",") if m.strip()]
    ordered = [{"role": "primary", "model": primary}] + [
        {"role": f"fallback {i + 1}", "model": m} for i, m in enumerate(chain)
    ]
    narrative = os.getenv("LLM_NARRATIVE_MODEL", "").strip()
    if narrative:
        ordered.append({"role": "narrative", "model": narrative})
    return [item for item in ordered if item["model"]]


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: float = 5) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _check_gemini(model: str) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"ok": False, "detail": "GEMINI_API_KEY not set"}
    name = model.split("/", 1)[-1]
    try:
        _http_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{name}",
            {"x-goog-api-key": key},
        )
        return {"ok": True, "detail": "model available (daily quota not checked)"}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "detail": f"HTTP {exc.code}"}
    except Exception as exc:  # network, DNS
        return {"ok": False, "detail": type(exc).__name__}


def _check_ollama(model: str) -> dict[str, Any]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    name = model.split("/", 1)[-1]
    try:
        tags = _http_json(f"{host}/api/tags", timeout=3)
    except Exception as exc:
        return {"ok": False, "detail": f"Ollama not running ({type(exc).__name__})"}
    names = {item.get("name") for item in tags.get("models", [])}
    if name in names or f"{name}:latest" in names:
        return {"ok": True, "detail": "running locally · $0 per call"}
    return {"ok": False, "detail": f"model not pulled (ollama pull {name})"}


def _check_openrouter(_model: str) -> dict[str, Any]:
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    if not key:
        return {"ok": False, "detail": "OPENROUTER_API_KEY not set"}
    try:
        data = _http_json(
            "https://openrouter.ai/api/v1/key", {"Authorization": f"Bearer {key}"}
        ).get("data", {})
    except urllib.error.HTTPError as exc:
        return {"ok": False, "detail": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"ok": False, "detail": type(exc).__name__}
    remaining = data.get("limit_remaining")
    if remaining is not None and float(remaining) <= 0.01:
        return {"ok": False, "detail": f"no credit left (${float(remaining):.2f})", "credit_usd": remaining}
    detail = f"credit ${float(remaining):.2f}" if remaining is not None else "key valid"
    return {"ok": True, "detail": detail, "credit_usd": remaining}


_CHECKS = {"gemini": _check_gemini, "ollama": _check_ollama, "openrouter": _check_openrouter}


def provider_status(force: bool = False) -> dict[str, Any]:
    if not force and _CACHE["value"] and time.time() - _CACHE["at"] < TTL_SECONDS:
        return _CACHE["value"]
    providers = []
    for item in _models_in_order():
        provider = provider_of(item["model"], os.getenv("LLM_PROVIDER", ""))
        check = _CHECKS.get(provider)
        result = check(item["model"]) if check else {"ok": False, "detail": "unknown provider"}
        providers.append({**item, "provider": provider, **result})
    cache_dir = decision_cache._directory()
    value = {
        "mode": os.getenv("LLM_MODE", "live"),
        "providers": providers,
        "cache": {
            "enabled": decision_cache.enabled(),
            "entries": len(list(cache_dir.glob("*.json"))) if cache_dir.is_dir() else 0,
        },
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _CACHE.update({"at": time.time(), "value": value})
    return value
