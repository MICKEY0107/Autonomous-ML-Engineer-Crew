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


DecisionSource = Literal["llm", "secondary_llm", "cache", "fallback"]


def crewai_llm() -> Any:
    """Load the heavyweight CrewAI provider only when a live decision is needed."""

    from orchestrator.llm_provider import crewai_llm as build_llm

    return build_llm()


def _record_llm_usage(llm: Any, *, stage: str, source: str, elapsed: float) -> None:
    from orchestrator import usage

    try:
        metrics = llm.get_token_usage_summary()
    except Exception:
        return
    usage.record(
        stage=stage,
        agent="Model Selection Agent (ADK)",
        framework="adk",
        provider=str(getattr(llm, "provider", "") or ""),
        model=str(getattr(llm, "model", "") or ""),
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
        requests=metrics.successful_requests,
        latency_seconds=round(elapsed, 3),
        source=source,
    )


def _call_live_llm(prompt: str, *, stage: str = "decision") -> tuple[Any, str]:
    """Try each provider in the chain once (no retry storms).

    Order: primary -> first fallback -> any further LLM_FALLBACK_CHAIN entries.
    Returns (response, source) where source is ``llm`` or ``secondary_llm``.
    """

    import time

    from orchestrator import llm_provider

    def candidates():
        yield "llm", crewai_llm
        yield "secondary_llm", llm_provider.fallback_crewai_llm
        for extra in llm_provider.extra_fallback_llms():
            yield "secondary_llm", (lambda built=extra: built)

    last_error: Exception | None = None
    for source, factory in candidates():
        started = time.perf_counter()
        try:
            llm = factory()
            if llm is None:
                continue
            response = llm.call(prompt)
        except Exception as exc:
            last_error = exc
            continue
        _record_llm_usage(llm, stage=stage, source=source, elapsed=time.perf_counter() - started)
        return response, source
    raise last_error or RuntimeError("no LLM provider configured")


def _from_cache(stage: str, inputs: dict[str, Any], model: type[BaseModel], agent: str, framework: str):
    """Return a cached, re-validated decision labelled source='cache', or None."""

    from orchestrator import decision_cache, usage

    cached = decision_cache.get(stage, inputs)
    if cached is None:
        return None
    try:
        decision = model.model_validate({**cached, "source": "cache"})
    except Exception:
        return None
    usage.record(stage=stage, agent=agent, framework=framework, source="cache")
    return decision


def _to_cache(stage: str, inputs: dict[str, Any], decision: BaseModel) -> None:
    from orchestrator import decision_cache

    decision_cache.put(stage, inputs, decision.model_dump(exclude={"source"}))


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


def _json_value(response: Any) -> Any:
    """Parse an LLM reply that may be fenced and may be an object or a list."""

    rendered = str(response).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", rendered, re.DOTALL)
    if fenced:
        rendered = fenced.group(1).strip()
    if rendered.startswith("["):
        return json.loads(rendered[: rendered.rfind("]") + 1])
    return _json_object(rendered)


def _number_like(value: Any) -> Any:
    """'0.1' -> 0.1, '200' -> 200, 'null'/'None' -> None; anything else unchanged."""

    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in {"null", "none", ""}:
            return None
        try:
            number = float(text)
        except ValueError:
            return value
        return int(number) if number.is_integer() and "." not in text else number
    if isinstance(value, float) and value.is_integer():
        return int(value)  # 100.0 -> 100 for integer-only parameters
    return value


def _normalise_search_plan(raw: Any) -> dict[str, Any]:
    """Repair common *format* slips of smaller models; bounds are checked after.

    Accepts a bare list of candidates, numbers written as strings, and a
    per-candidate ``reasoning`` field. It never invents parameters or values.
    """

    if isinstance(raw, list):
        raw = {"candidates": raw}
    if not isinstance(raw, dict):
        raise ValueError("model-search plan must be a JSON object or list")
    notes: list[str] = []
    candidates = []
    for item in raw.get("candidates", []):
        if not isinstance(item, dict):
            raise ValueError("each candidate must be an object")
        if item.get("reasoning"):
            notes.append(f"{item.get('name')}: {item['reasoning']}")
        parameters = item.get("parameters") or {}
        candidates.append(
            {
                "name": str(item.get("name", "")).strip(),
                "parameters": {k: _number_like(v) for k, v in parameters.items()},
            }
        )
    reasoning = str(raw.get("reasoning") or " ".join(notes) or "").strip()
    return {"candidates": candidates, "reasoning": reasoning[:2_000] or "LLM-configured search."}


def _normalise_choice(raw: Any) -> dict[str, Any]:
    """Repair format slips in a model choice (tradeoffs as objects, numbers as text)."""

    if not isinstance(raw, dict):
        raise ValueError("model choice must be a JSON object")
    tradeoffs = []
    for item in raw.get("tradeoffs") or []:
        if isinstance(item, dict):
            item = item.get("description") or json.dumps(item, sort_keys=True)
        tradeoffs.append(str(item)[:300])
    return {
        "selected_model_name": str(raw.get("selected_model_name", "")).strip(),
        "reasoning": str(raw.get("reasoning", "")).strip()[:2_000],
        "tradeoffs": tradeoffs[:6],
        "confidence": _number_like(raw.get("confidence", 0.5)),
    }


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
            from orchestrator.crew_agents import run_crew_decision

            available = set(schema) - {target_column}
            cached = _from_cache(
                "data_prep_plan", safe_input, DataPrepPlan, "Data Preparation Engineer", "crewai"
            )
            if cached is not None and set(cached.additional_exclusions) <= available:
                update_span_io(span, output={**cached.model_dump(), "fallback_used": False})
                return cached

            def check_plan(candidate: DataPrepPlan) -> None:
                proposed = set(candidate.additional_exclusions)
                if not proposed.issubset(available):
                    raise ValueError(
                        "additional_exclusions may only name these columns: "
                        f"{sorted(available)}"
                    )
                if len(proposed) >= len(available):
                    raise ValueError("the plan would remove every feature")

            # Real CrewAI execution: Agent + Task(output_pydantic, guardrail) + Crew.
            plan, source = run_crew_decision(
                stage="data_prep_plan",
                role="Data Preparation Engineer",
                goal=(
                    "Decide how this tabular dataset should be cleaned and split so "
                    "the model generalises without leakage."
                ),
                backstory=(
                    "You are a careful data engineer. You only see schema-level "
                    "metadata, never raw rows, and you justify every action."
                ),
                task_description=prompt,
                expected_output=(
                    "A DataPrepPlan JSON object: additional_exclusions, test_size, "
                    "numeric_imputation, categorical_imputation, reasoning."
                ),
                output_model=DataPrepPlan,
                validator=check_plan,
            )
            plan = plan.model_copy(update={"source": source})
            _to_cache("data_prep_plan", safe_input, plan)
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
        "catalog exactly once. Return ONE JSON object only, shaped exactly like "
        '{"candidates": [{"name": "<catalog name>", "parameters": {"<param>": '
        '<number or null>}}, ...], "reasoning": "<why these settings>"}. '
        "Use numbers, not strings, for parameter values; use only the parameters "
        "listed for each model and respect every bound. "
        f"Context: {json.dumps(safe_input, sort_keys=True)}. "
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
            cached = _from_cache(
                "model_search_plan", safe_input, ModelSearchPlan,
                "Model Selection Agent (ADK)", "adk",
            )
            if cached is not None:
                _validate_search_plan(cached, task_type)
                update_span_io(span, output={**cached.model_dump(), "fallback_used": False})
                return cached
            response, source = _call_live_llm(prompt, stage="model_search_plan")
            plan = ModelSearchPlan.model_validate(_normalise_search_plan(_json_value(response)))
            _validate_search_plan(plan, task_type)
            plan = plan.model_copy(update={"source": source})
            _to_cache("model_search_plan", safe_input, plan)
            update_span_io(span, output={**plan.model_dump(), "fallback_used": False})
            return plan
        except Exception as exc:
            from orchestrator import usage

            # Recorded so labels and reports show this decision used the policy.
            usage.record(
                stage="model_search_plan",
                agent="Model Selection Agent (ADK)",
                framework="adk",
                source="fallback",
            )
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
        "and the prediction goal. Return ONE JSON object only, shaped exactly like "
        '{"selected_model_name": "<one eligible name>", "reasoning": "<why>", '
        '"tradeoffs": ["<short sentence>", ...], "confidence": 0.8}. '
        "You cannot choose a model outside the eligible list. Evidence: "
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
            cached = _from_cache(
                "model_choice", safe_input, ModelChoice, "Model Selection Agent (ADK)", "adk"
            )
            if cached is not None and cached.selected_model_name in eligible:
                update_span_io(span, output={**cached.model_dump(), "fallback_used": False})
                return cached
            response, source = _call_live_llm(prompt, stage="model_choice")
            choice = ModelChoice.model_validate(_normalise_choice(_json_value(response)))
            choice = choice.model_copy(update={"source": source})
            if choice.selected_model_name not in eligible:
                raise ValueError("LLM selected a model outside the quality band")
            if not math.isfinite(choice.confidence):
                raise ValueError("LLM confidence must be finite")
            _to_cache("model_choice", safe_input, choice)
            update_span_io(span, output={**choice.model_dump(), "fallback_used": False})
            return choice
        except Exception as exc:
            from orchestrator import usage

            # Recorded so labels and reports show this decision used the policy.
            usage.record(
                stage="model_choice",
                agent="Model Selection Agent (ADK)",
                framework="adk",
                source="fallback",
            )
            update_span_io(
                span,
                output={
                    **fallback.model_dump(),
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
            return fallback
