"""Typed, executable LLM decisions for the autonomous ML pipeline.

The LLM receives row-free metadata and proposes bounded actions.  Deterministic
code validates those actions before execution, so an invalid or unavailable LLM
cannot bypass data-safety or model-quality constraints.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Literal

from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.telemetry import traced_span, update_span_io


DecisionSource = Literal["llm", "fallback"]


def crewai_llm() -> Any:
    """Load the heavyweight CrewAI provider only when a live decision is needed."""

    from orchestrator.llm_provider import crewai_llm as build_llm

    return build_llm()


def _call_live_llm(prompt: str) -> Any:
    """Call the primary LLM, then an independent provider on API failure."""

    try:
        return crewai_llm().call(prompt)
    except Exception:
        from orchestrator.llm_provider import fallback_crewai_llm

        fallback = fallback_crewai_llm()
        if fallback is None:
            raise
        return fallback.call(prompt)


class DataPrepPlan(BaseModel):
    """Bounded data-preparation actions that deterministic code can execute."""

    model_config = ConfigDict(extra="forbid")

    additional_exclusions: list[str] = Field(default_factory=list)
    test_size: float = Field(ge=0.15, le=0.30)
    numeric_imputation: Literal["median", "mean"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    reasoning: str = Field(min_length=1, max_length=2_000)
    source: DecisionSource = "llm"


class ModelChoice(BaseModel):
    """Auditable LLM choice among already-trained, quality-eligible models."""

    model_config = ConfigDict(extra="forbid")

    selected_model_name: str = Field(min_length=1)
    reasoning: str = Field(min_length=1, max_length=2_000)
    tradeoffs: list[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0, le=1)
    source: DecisionSource = "llm"


class CandidatePlan(BaseModel):
    """One allowlisted estimator and its bounded, executable parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    parameters: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class ModelSearchPlan(BaseModel):
    """LLM-proposed candidate configuration for one training attempt."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidatePlan] = Field(min_length=3, max_length=3)
    reasoning: str = Field(min_length=1, max_length=2_000)
    source: DecisionSource = "llm"


def _offline() -> bool:
    return os.getenv("LLM_MODE", "live").strip().casefold() != "live"


def _json_object(response: Any) -> dict[str, Any]:
    rendered = str(response).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", rendered, re.DOTALL)
    if fenced:
        rendered = fenced.group(1)
    start = rendered.find("{")
    end = rendered.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM decision did not contain a JSON object")
    decoded = json.loads(rendered[start : end + 1])
    if not isinstance(decoded, dict):
        raise ValueError("LLM decision must be a JSON object")
    return decoded


def plan_data_prep(
    *,
    schema: dict[str, str],
    null_counts: dict[str, int],
    leakage_candidates: list[str],
    row_count: int,
    target_column: str,
    prediction_goal: str,
    requested_test_size: float,
) -> DataPrepPlan:
    """Ask the LLM for a row-free preparation plan and validate its actions."""

    fallback = DataPrepPlan(
        additional_exclusions=[],
        test_size=min(0.30, max(0.15, requested_test_size)),
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        reasoning=(
            "Used the safe preparation policy: exclude detected leakage fields, "
            "retain the requested split, and impute from training-only statistics."
        ),
        source="fallback",
    )
    safe_input = {
        "schema": schema,
        "null_counts": null_counts,
        "leakage_candidates": leakage_candidates,
        "row_count": row_count,
        "target_column": target_column,
        "prediction_goal": prediction_goal,
        "requested_test_size": requested_test_size,
    }
    prompt = (
        "Act as the data-preparation decision agent. Return JSON only with keys: "
        "additional_exclusions (column-name array), test_size (0.15 to 0.30), "
        "numeric_imputation ('median' or 'mean'), categorical_imputation "
        "('most_frequent' or 'constant'), reasoning (concise). Never exclude the "
        "target. Use only this row-free metadata: "
        f"{json.dumps(safe_input, sort_keys=True)}"
    )
    with traced_span(
        "llm.decision.data_prep_plan",
        kind=SpanKind.CLIENT,
        attributes={"ml.reasoning.row_safe": True},
    ) as span:
        update_span_io(span, input=safe_input)
        try:
            if _offline():
                raise RuntimeError("LLM_MODE is not live")
            plan = DataPrepPlan.model_validate(_json_object(_call_live_llm(prompt)))
            available = set(schema) - {target_column}
            proposed = set(plan.additional_exclusions)
            if not proposed.issubset(available):
                raise ValueError("data-prep plan referenced an unknown or target column")
            if len(proposed) >= len(available):
                raise ValueError("data-prep plan would remove every feature")
            update_span_io(span, output={**plan.model_dump(), "fallback_used": False})
            return plan
        except Exception as exc:
            update_span_io(
                span,
                output={
                    **fallback.model_dump(),
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
            return fallback


def _eligible_model_names(
    candidates: list[dict[str, Any]],
    *,
    maximize: bool,
) -> set[str]:
    values = [float(candidate["metric_value"]) for candidate in candidates]
    best = max(values) if maximize else min(values)
    if maximize:
        tolerance = 0.01
        return {
            str(candidate["name"])
            for candidate in candidates
            if float(candidate["metric_value"]) >= best - tolerance
        }
    tolerance = max(abs(best) * 0.01, 1e-12)
    return {
        str(candidate["name"])
        for candidate in candidates
        if float(candidate["metric_value"]) <= best + tolerance
    }


def _fallback_search_plan(task_type: str, retry_attempt: int) -> ModelSearchPlan:
    if task_type == "classification":
        return ModelSearchPlan(
            candidates=[
                CandidatePlan(
                    name="logistic_regression",
                    parameters={"C": (1.0, 0.5, 2.0)[retry_attempt], "max_iter": 1000},
                ),
                CandidatePlan(
                    name="random_forest",
                    parameters={
                        "n_estimators": 100 + 50 * retry_attempt,
                        "max_depth": None,
                        "class_weight": "balanced",
                    },
                ),
                CandidatePlan(
                    name="gradient_boosting",
                    parameters={
                        "learning_rate": (0.1, 0.05, 0.2)[retry_attempt],
                        "n_estimators": 100,
                        "max_depth": 3,
                    },
                ),
            ],
            reasoning="Used the safe bounded candidate-search policy.",
            source="fallback",
        )
    return ModelSearchPlan(
        candidates=[
            CandidatePlan(name="linear_regression", parameters={}),
            CandidatePlan(
                name="random_forest",
                parameters={
                    "n_estimators": 100 + 50 * retry_attempt,
                    "max_depth": None,
                },
            ),
            CandidatePlan(
                name="ridge",
                parameters={"alpha": (1.0, 0.1, 10.0)[retry_attempt]},
            ),
        ],
        reasoning="Used the safe bounded candidate-search policy.",
        source="fallback",
    )


def _bounded_number(
    parameters: dict[str, Any],
    name: str,
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
) -> None:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} is outside the executable range")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _validate_search_plan(plan: ModelSearchPlan, task_type: str) -> None:
    expected = (
        {"logistic_regression", "random_forest", "gradient_boosting"}
        if task_type == "classification"
        else {"linear_regression", "random_forest", "ridge"}
    )
    by_name = {candidate.name: candidate.parameters for candidate in plan.candidates}
    if set(by_name) != expected:
        raise ValueError("model-search plan must configure every allowlisted candidate once")

    if task_type == "classification":
        logistic = by_name["logistic_regression"]
        if set(logistic) != {"C", "max_iter"}:
            raise ValueError("invalid logistic-regression parameter set")
        _bounded_number(logistic, "C", 0.01, 100)
        _bounded_number(logistic, "max_iter", 200, 3_000, integer=True)

        forest = by_name["random_forest"]
        if set(forest) != {"n_estimators", "max_depth", "class_weight"}:
            raise ValueError("invalid classification random-forest parameter set")
        _bounded_number(forest, "n_estimators", 50, 500, integer=True)
        if forest["max_depth"] is not None:
            _bounded_number(forest, "max_depth", 2, 50, integer=True)
        if forest["class_weight"] not in {"balanced", None}:
            raise ValueError("class_weight must be balanced or null")

        boosting = by_name["gradient_boosting"]
        if set(boosting) != {"learning_rate", "n_estimators", "max_depth"}:
            raise ValueError("invalid gradient-boosting parameter set")
        _bounded_number(boosting, "learning_rate", 0.01, 0.5)
        _bounded_number(boosting, "n_estimators", 50, 500, integer=True)
        _bounded_number(boosting, "max_depth", 1, 10, integer=True)
        return

    if by_name["linear_regression"]:
        raise ValueError("linear_regression does not accept planned parameters")
    forest = by_name["random_forest"]
    if set(forest) != {"n_estimators", "max_depth"}:
        raise ValueError("invalid regression random-forest parameter set")
    _bounded_number(forest, "n_estimators", 50, 500, integer=True)
    if forest["max_depth"] is not None:
        _bounded_number(forest, "max_depth", 2, 50, integer=True)
    ridge = by_name["ridge"]
    if set(ridge) != {"alpha"}:
        raise ValueError("invalid ridge parameter set")
    _bounded_number(ridge, "alpha", 0.001, 1_000)


def plan_model_search(
    *,
    task_type: str,
    row_count: int,
    feature_count: int,
    prediction_goal: str,
    retry_attempt: int,
    retry_feedback: str | None,
) -> ModelSearchPlan:
    """Produce a bounded hyperparameter plan that the ADK agent will execute."""

    fallback = _fallback_search_plan(task_type, retry_attempt)
    safe_input = {
        "task_type": task_type,
        "row_count": row_count,
        "feature_count": feature_count,
        "prediction_goal": prediction_goal,
        "retry_attempt": retry_attempt,
        "retry_feedback": retry_feedback,
    }
    catalog = (
        {
            "logistic_regression": {"C": "0.01..100", "max_iter": "200..3000 integer"},
            "random_forest": {
                "n_estimators": "50..500 integer",
                "max_depth": "null or 2..50 integer",
                "class_weight": "balanced or null",
            },
            "gradient_boosting": {
                "learning_rate": "0.01..0.5",
                "n_estimators": "50..500 integer",
                "max_depth": "1..10 integer",
            },
        }
        if task_type == "classification"
        else {
            "linear_regression": {},
            "random_forest": {
                "n_estimators": "50..500 integer",
                "max_depth": "null or 2..50 integer",
            },
            "ridge": {"alpha": "0.001..1000"},
        }
    )
    prompt = (
        "Act as the model-search planning agent. Configure every model in the "
        "catalog exactly once. Return JSON only with candidates (three objects, "
        "each containing name and parameters) and reasoning. Respect every "
        f"parameter bound. Context: {json.dumps(safe_input, sort_keys=True)}. "
        f"Catalog: {json.dumps(catalog, sort_keys=True)}"
    )
    with traced_span(
        "llm.decision.model_search_plan",
        kind=SpanKind.CLIENT,
        attributes={"ml.reasoning.row_safe": True},
    ) as span:
        update_span_io(span, input={**safe_input, "catalog": catalog})
        try:
            if _offline():
                raise RuntimeError("LLM_MODE is not live")
            plan = ModelSearchPlan.model_validate(_json_object(_call_live_llm(prompt)))
            _validate_search_plan(plan, task_type)
            update_span_io(span, output={**plan.model_dump(), "fallback_used": False})
            return plan
        except Exception as exc:
            update_span_io(
                span,
                output={
                    **fallback.model_dump(),
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
            return fallback


def choose_model(
    *,
    candidates: list[dict[str, Any]],
    task_type: str,
    primary_metric: str,
    maximize: bool,
    prediction_goal: str,
    retry_feedback: str | None,
) -> ModelChoice:
    """Let the LLM choose among candidates inside a deterministic quality band."""

    if not candidates:
        raise ValueError("at least one model candidate is required")
    eligible = _eligible_model_names(candidates, maximize=maximize)
    eligible_candidates = [
        candidate for candidate in candidates if str(candidate["name"]) in eligible
    ]
    fallback_candidate = min(
        eligible_candidates,
        key=lambda candidate: (
            int(candidate.get("training_time_ms", 0)),
            str(candidate["name"]),
        ),
    )
    fallback = ModelChoice(
        selected_model_name=str(fallback_candidate["name"]),
        reasoning=(
            "Selected the fastest candidate inside the deterministic one-percent "
            "quality band because the live decision model was unavailable or invalid."
        ),
        tradeoffs=["Quality eligibility was enforced before speed was considered."],
        confidence=0.5,
        source="fallback",
    )
    safe_input = {
        "task_type": task_type,
        "prediction_goal": prediction_goal,
        "primary_metric": primary_metric,
        "metric_direction": "maximize" if maximize else "minimize",
        "eligible_model_names": sorted(eligible),
        "candidates": candidates,
        "retry_feedback": retry_feedback,
    }
    prompt = (
        "Act as the model-selection decision agent. Choose exactly one model from "
        "eligible_model_names using measured quality, training time, robustness, "
        "and the prediction goal. Return JSON only with selected_model_name, "
        "reasoning, tradeoffs (array), and confidence (0 to 1). You cannot choose "
        "a model outside the eligible list. Evidence: "
        f"{json.dumps(safe_input, sort_keys=True)}"
    )
    with traced_span(
        "llm.decision.model_choice",
        kind=SpanKind.CLIENT,
        attributes={"ml.reasoning.row_safe": True},
    ) as span:
        update_span_io(span, input=safe_input)
        try:
            if _offline():
                raise RuntimeError("LLM_MODE is not live")
            choice = ModelChoice.model_validate(_json_object(_call_live_llm(prompt)))
            if choice.selected_model_name not in eligible:
                raise ValueError("LLM selected a model outside the quality band")
            if not math.isfinite(choice.confidence):
                raise ValueError("LLM confidence must be finite")
            update_span_io(span, output={**choice.model_dump(), "fallback_used": False})
            return choice
        except Exception as exc:
            update_span_io(
                span,
                output={
                    **fallback.model_dump(),
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
            return fallback
