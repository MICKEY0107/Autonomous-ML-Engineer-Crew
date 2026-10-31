"""Regression tests for the issues found in the end-to-end review (31 Oct)."""

from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from agents import deployment
from orchestrator import insights, usage
from orchestrator.agentic_policy import choose_model


class _FakeLlm:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def call(self, _prompt: str) -> str:
        self.calls += 1
        return json.dumps(self.payload)


class _FailingLlm:
    def call(self, _prompt: str) -> str:
        raise RuntimeError("provider down")


BAND = [
    {"name": "logistic_regression", "metric_value": 0.930, "training_time_ms": 20},
    {"name": "random_forest", "metric_value": 0.935, "training_time_ms": 80},
    {"name": "gradient_boosting", "metric_value": 0.800, "training_time_ms": 90},
]


def _choose(candidates, retry_feedback=None):
    return choose_model(
        candidates=candidates,
        task_type="classification",
        primary_metric="f1_weighted",
        maximize=True,
        prediction_goal="classify species",
        retry_feedback=retry_feedback,
    )


def test_fallback_prefers_speed_first_time_and_quality_on_retry() -> None:
    # conftest runs offline, so the deterministic rule decides.
    assert _choose(BAND).selected_model_name == "logistic_regression"
    retry = _choose(BAND, retry_feedback="F1 0.93 is below 0.99; improve quality.")
    assert retry.selected_model_name == "random_forest"
    assert "best-scoring" in retry.reasoning


def test_model_choice_cache_ignores_training_time(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("LLM_CACHE", "on")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    live = _FakeLlm(
        {"selected_model_name": "random_forest", "reasoning": "Robust.", "tradeoffs": [], "confidence": 0.8}
    )
    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: live)
    assert _choose(BAND).source == "llm"

    # Same scores, different timings (as on every real rerun) -> cache hit.
    retimed = [{**item, "training_time_ms": item["training_time_ms"] + 7} for item in BAND]
    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: _FailingLlm())
    monkeypatch.setattr("orchestrator.llm_provider.fallback_crewai_llm", lambda: None)
    usage.reset()
    again = _choose(retimed)
    assert again.source == "cache" and again.selected_model_name == "random_forest"


def test_forced_fail_threshold_follows_metric_direction() -> None:
    import orchestrator_api

    assert orchestrator_api._forced_fail_threshold("classification") == 0.99
    # RMSE is lower-is-better: only a perfect fit (0) would pass.
    assert orchestrator_api._forced_fail_threshold("regression") == 0.0


def test_exhausted_note_names_the_closest_attempt() -> None:
    def attempt(number: int, model: str, rmse: float) -> dict:
        return {
            "attempt": number,
            "selected_model": model,
            "evaluation": {
                "primary_metric": "rmse",
                "held_out_metrics": {"rmse": rmse},
                "threshold": 1.0,
                "direction": "minimize",
            },
        }

    note = insights._exhausted_note(
        [attempt(1, "ridge", 120.0), attempt(2, "random_forest", 95.0), attempt(3, "ridge", 130.0)],
        "ridge",
    )
    assert "closest was random forest on attempt 2" in note
    assert "final attempt" in note


def test_feature_schema_does_not_offer_nan_as_a_category(tmp_path) -> None:
    csv = tmp_path / "train.csv"
    pd.DataFrame(
        {"plan": ["basic", None, None, None, "pro"], "spend": [1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)
    plan = next(item for item in deployment._feature_schema(csv, ["plan", "spend"]) if item["name"] == "plan")
    assert "nan" not in plan["categories"]
    assert plan["example"] in {"basic", "pro"}


def test_second_run_is_refused_while_one_is_starting() -> None:
    import orchestrator_api

    client = TestClient(orchestrator_api.app)
    orchestrator_api._starting = True
    try:
        response = client.post(
            "/api/pipeline/run",
            files={"csv_file": ("x.csv", b"a,b\n1,2\n", "text/csv")},
            data={"target_column": "b", "prediction_goal": "predict b"},
        )
    finally:
        orchestrator_api._starting = False
    assert response.status_code == 409


def test_deployment_without_model_file_is_404(tmp_path, monkeypatch) -> None:
    import orchestrator_api

    run_hex = "f" * 32
    folder = tmp_path / run_hex
    folder.mkdir()
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(orchestrator_api, "bundle_dir", lambda value: tmp_path / value)
    client = TestClient(orchestrator_api.app)
    assert client.get(f"/api/deployments/{run_hex}").status_code == 404
