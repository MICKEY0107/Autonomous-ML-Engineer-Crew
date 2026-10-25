"""Tests for typed LLM decisions and deterministic execution guardrails."""

from __future__ import annotations

import json

from orchestrator.agentic_policy import (
    choose_model,
    plan_data_prep,
    plan_model_search,
)


class _FakeLlm:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def call(self, _prompt: str) -> str:
        return json.dumps(self.payload)


class _FailingLlm:
    def call(self, _prompt: str) -> str:
        raise RuntimeError("primary provider unavailable")


def test_data_prep_plan_is_typed_and_executable(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FakeLlm(
            {
                "additional_exclusions": ["record_id"],
                "test_size": 0.25,
                "numeric_imputation": "mean",
                "categorical_imputation": "constant",
                "reasoning": "Exclude the identifier and reserve a balanced holdout.",
            }
        ),
    )

    plan = plan_data_prep(
        schema={"record_id": "int64", "x": "float64", "target": "str"},
        null_counts={"record_id": 0, "x": 2, "target": 0},
        leakage_candidates=["record_id"],
        row_count=100,
        target_column="target",
        prediction_goal="classify target",
        requested_test_size=0.2,
    )

    assert plan.source == "llm"
    assert plan.additional_exclusions == ["record_id"]
    assert plan.numeric_imputation == "mean"
    assert plan.test_size == 0.25


def test_data_prep_plan_fails_closed_on_unknown_column(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FakeLlm(
            {
                "additional_exclusions": ["not_a_column"],
                "test_size": 0.25,
                "numeric_imputation": "mean",
                "categorical_imputation": "constant",
                "reasoning": "Invalid proposal.",
            }
        ),
    )

    plan = plan_data_prep(
        schema={"x": "float64", "target": "str"},
        null_counts={"x": 0, "target": 0},
        leakage_candidates=[],
        row_count=100,
        target_column="target",
        prediction_goal="classify target",
        requested_test_size=0.2,
    )

    assert plan.source == "fallback"
    assert plan.additional_exclusions == []


def test_llm_chooses_between_quality_eligible_models(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FakeLlm(
            {
                "selected_model_name": "random_forest",
                "reasoning": "The tied ensemble is preferred for nonlinear robustness.",
                "tradeoffs": ["Slower than logistic regression."],
                "confidence": 0.82,
            }
        ),
    )
    candidates = [
        {"name": "logistic_regression", "metric_value": 0.93, "training_time_ms": 20},
        {"name": "random_forest", "metric_value": 0.93, "training_time_ms": 80},
        {"name": "gradient_boosting", "metric_value": 0.84, "training_time_ms": 100},
    ]

    choice = choose_model(
        candidates=candidates,
        task_type="classification",
        primary_metric="f1_weighted",
        maximize=True,
        prediction_goal="classify species",
        retry_feedback=None,
    )

    assert choice.source == "llm"
    assert choice.selected_model_name == "random_forest"


def test_model_choice_rejects_quality_band_violation(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FakeLlm(
            {
                "selected_model_name": "gradient_boosting",
                "reasoning": "Choose the weakest candidate.",
                "tradeoffs": [],
                "confidence": 0.9,
            }
        ),
    )
    candidates = [
        {"name": "logistic_regression", "metric_value": 0.93, "training_time_ms": 20},
        {"name": "random_forest", "metric_value": 0.93, "training_time_ms": 80},
        {"name": "gradient_boosting", "metric_value": 0.84, "training_time_ms": 100},
    ]

    choice = choose_model(
        candidates=candidates,
        task_type="classification",
        primary_metric="f1_weighted",
        maximize=True,
        prediction_goal="classify species",
        retry_feedback=None,
    )

    assert choice.source == "fallback"
    assert choice.selected_model_name == "logistic_regression"


def test_llm_model_search_plan_is_typed_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FakeLlm(
            {
                "candidates": [
                    {
                        "name": "logistic_regression",
                        "parameters": {"C": 0.5, "max_iter": 1200},
                    },
                    {
                        "name": "random_forest",
                        "parameters": {
                            "n_estimators": 240,
                            "max_depth": 12,
                            "class_weight": "balanced",
                        },
                    },
                    {
                        "name": "gradient_boosting",
                        "parameters": {
                            "learning_rate": 0.07,
                            "n_estimators": 180,
                            "max_depth": 2,
                        },
                    },
                ],
                "reasoning": "Use regularized linear and bounded ensemble candidates.",
            }
        ),
    )

    plan = plan_model_search(
        task_type="classification",
        row_count=120,
        feature_count=4,
        prediction_goal="classify species",
        retry_attempt=0,
        retry_feedback=None,
    )

    assert plan.source == "llm"
    assert [candidate.name for candidate in plan.candidates] == [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ]
    assert plan.candidates[1].parameters["n_estimators"] == 240


def test_model_search_plan_rejects_out_of_bounds_parameters(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FakeLlm(
            {
                "candidates": [
                    {
                        "name": "logistic_regression",
                        "parameters": {"C": 0.5, "max_iter": 1200},
                    },
                    {
                        "name": "random_forest",
                        "parameters": {
                            "n_estimators": 50_000,
                            "max_depth": None,
                            "class_weight": "balanced",
                        },
                    },
                    {
                        "name": "gradient_boosting",
                        "parameters": {
                            "learning_rate": 0.07,
                            "n_estimators": 180,
                            "max_depth": 2,
                        },
                    },
                ],
                "reasoning": "This unsafe plan must not execute.",
            }
        ),
    )

    plan = plan_model_search(
        task_type="classification",
        row_count=120,
        feature_count=4,
        prediction_goal="classify species",
        retry_attempt=0,
        retry_feedback=None,
    )

    assert plan.source == "fallback"
    assert plan.candidates[1].parameters["n_estimators"] == 100


def test_agentic_decision_uses_secondary_llm_on_primary_api_failure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(
        "orchestrator.agentic_policy.crewai_llm",
        lambda: _FailingLlm(),
    )
    monkeypatch.setattr(
        "orchestrator.llm_provider.fallback_crewai_llm",
        lambda: _FakeLlm(
            {
                "additional_exclusions": ["record_id"],
                "test_size": 0.2,
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "reasoning": "The alternate provider supplied a safe plan.",
            }
        ),
    )

    plan = plan_data_prep(
        schema={"record_id": "int64", "x": "float64", "target": "str"},
        null_counts={"record_id": 0, "x": 0, "target": 0},
        leakage_candidates=["record_id"],
        row_count=100,
        target_column="target",
        prediction_goal="classify target",
        requested_test_size=0.2,
    )

    assert plan.source == "llm"
    assert plan.additional_exclusions == ["record_id"]
