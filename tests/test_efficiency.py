"""Decision cache, provider chain and per-provider cost accounting."""

from __future__ import annotations

import json

from orchestrator import usage
from orchestrator.agentic_policy import choose_model
from orchestrator.llm_provider import _chain_models, provider_of


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


CANDIDATES = [
    {"name": "logistic_regression", "metric_value": 0.93, "training_time_ms": 20},
    {"name": "random_forest", "metric_value": 0.93, "training_time_ms": 80},
    {"name": "gradient_boosting", "metric_value": 0.80, "training_time_ms": 90},
]


def _choose():
    return choose_model(
        candidates=CANDIDATES,
        task_type="classification",
        primary_metric="f1_weighted",
        maximize=True,
        prediction_goal="classify species",
        retry_feedback=None,
    )


def test_cached_decision_is_reused_and_labelled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("LLM_CACHE", "on")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    live = _FakeLlm(
        {
            "selected_model_name": "random_forest",
            "reasoning": "Robust ensemble.",
            "tradeoffs": [],
            "confidence": 0.8,
        }
    )
    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: live)

    first = _choose()
    assert first.source == "llm"
    assert live.calls == 1

    # Same question again: no LLM call (the provider is now down), cached answer.
    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: _FailingLlm())
    monkeypatch.setattr("orchestrator.llm_provider.fallback_crewai_llm", lambda: None)
    usage.reset()
    second = _choose()
    assert second.source == "cache"
    assert second.selected_model_name == "random_forest"
    assert usage.summary()["cache_hits"] == 1


def test_cache_off_always_asks_the_llm(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("LLM_CACHE", "off")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    live = _FakeLlm(
        {
            "selected_model_name": "logistic_regression",
            "reasoning": "Fastest tied model.",
            "tradeoffs": [],
            "confidence": 0.7,
        }
    )
    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: live)
    _choose()
    _choose()
    assert live.calls == 2
    assert not list(tmp_path.iterdir())


def test_chain_falls_through_every_provider_once(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    third = _FakeLlm(
        {
            "selected_model_name": "logistic_regression",
            "reasoning": "Local model answered.",
            "tradeoffs": [],
            "confidence": 0.6,
        }
    )
    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: _FailingLlm())
    monkeypatch.setattr("orchestrator.llm_provider.fallback_crewai_llm", lambda: _FailingLlm())
    monkeypatch.setattr("orchestrator.llm_provider.extra_fallback_llms", lambda: [third])
    choice = _choose()
    assert choice.source == "secondary_llm"
    assert third.calls == 1


def test_local_models_cost_nothing_and_cloud_prices_apply(monkeypatch) -> None:
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M", "1.0")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M", "2.0")
    monkeypatch.setenv("OPENROUTER_INPUT_COST_PER_1M", "3.0")
    assert usage.estimate_cost(1_000_000, 1_000_000, "ollama") == 0.0
    assert usage.estimate_cost(1_000_000, 1_000_000, "gemini") == 3.0
    assert usage.estimate_cost(1_000_000, 0, "openrouter") == 3.0


def test_provider_prefixes_and_chain_parsing(monkeypatch) -> None:
    assert provider_of("ollama/qwen2.5-coder:latest") == "ollama"
    assert provider_of("gemini/gemini-3.5-flash-lite") == "gemini"
    assert provider_of("openrouter/google/x") == "openrouter"
    monkeypatch.setenv(
        "LLM_FALLBACK_CHAIN", " ollama/qwen2.5-coder:latest , openrouter/google/x ,"
    )
    assert _chain_models() == ["ollama/qwen2.5-coder:latest", "openrouter/google/x"]


def test_small_model_format_slips_are_repaired_then_bounds_checked(monkeypatch) -> None:
    """Real replies from a local qwen2.5-coder model, which used to fall back."""

    from orchestrator.agentic_policy import plan_model_search

    monkeypatch.setenv("LLM_MODE", "live")
    reply = """```json
[
  {"name": "linear_regression", "parameters": {}, "reasoning": "Simple baseline."},
  {"name": "ridge", "parameters": {"alpha": "0.1"}, "reasoning": "Handles collinearity."},
  {"name": "random_forest", "parameters": {"n_estimators": "200", "max_depth": null}}
]
```"""

    class _Reply:
        def call(self, _prompt: str) -> str:
            return reply

    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: _Reply())
    plan = plan_model_search(
        task_type="regression",
        row_count=400,
        feature_count=9,
        prediction_goal="predict house sale price",
        retry_attempt=0,
        retry_feedback=None,
    )
    assert plan.source == "llm"
    params = {c.name: c.parameters for c in plan.candidates}
    assert params["ridge"] == {"alpha": 0.1}
    assert params["random_forest"] == {"n_estimators": 200, "max_depth": None}
    assert "collinearity" in plan.reasoning

    choice_reply = json.dumps(
        {
            "selected_model_name": "random_forest",
            "reasoning": "Tied on quality, more robust.",
            "tradeoffs": [{"model": "random_forest", "description": "Slower to train."}],
            "confidence": "0.9",
        }
    )

    class _ChoiceReply:
        def call(self, _prompt: str) -> str:
            return choice_reply

    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: _ChoiceReply())
    choice = _choose()
    assert choice.source == "llm"
    assert choice.tradeoffs == ["Slower to train."]
    assert choice.confidence == 0.9


def test_out_of_bounds_values_still_fall_back(monkeypatch) -> None:
    from orchestrator.agentic_policy import plan_model_search

    monkeypatch.setenv("LLM_MODE", "live")
    bad = json.dumps(
        {
            "candidates": [
                {"name": "linear_regression", "parameters": {}},
                {"name": "ridge", "parameters": {"alpha": "5000"}},
                {"name": "random_forest", "parameters": {"n_estimators": 100, "max_depth": None}},
            ],
            "reasoning": "alpha far too large",
        }
    )

    class _Bad:
        def call(self, _prompt: str) -> str:
            return bad

    monkeypatch.setattr("orchestrator.agentic_policy.crewai_llm", lambda: _Bad())
    monkeypatch.setattr("orchestrator.llm_provider.fallback_crewai_llm", lambda: None)
    plan = plan_model_search(
        task_type="regression",
        row_count=400,
        feature_count=9,
        prediction_goal="predict house sale price",
        retry_attempt=0,
        retry_feedback=None,
    )
    assert plan.source == "fallback"
