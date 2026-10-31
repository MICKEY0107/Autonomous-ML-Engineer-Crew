"""Optional Gemini reasoning for deterministic pipeline decisions.

These helpers never receive dataset rows and fail open to concise deterministic
explanations. Model training, evaluation gates, and retry routing remain fully
deterministic.
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry.trace import SpanKind

from orchestrator.llm_provider import crewai_llm, fallback_crewai_llm
from orchestrator.telemetry import traced_span, update_span_io


def _call_reasoner(
    *,
    span_name: str,
    prompt: str,
    safe_input: dict[str, Any],
    fallback: str,
) -> str:
    """Call one LLM with row-free metadata and return a safe fallback on error."""

    provider = os.getenv("LLM_PROVIDER", "not-configured").lower()
    model = os.getenv("CREWAI_MODEL", "not-configured")
    with traced_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "ml.reasoning.row_safe": True,
        },
    ) as span:
        update_span_io(span, input=safe_input)
        if os.getenv("LLM_MODE", "live").strip().casefold() != "live":
            update_span_io(
                span,
                output={
                    "reasoning": fallback,
                    "fallback_used": True,
                    "fallback_reason": "LLM_MODE is offline",
                },
            )
            return fallback
        try:
            import time

            from orchestrator import usage

            from orchestrator import llm_provider

            def candidates():
                # Narrative text may use a cheaper/local model first
                # (LLM_NARRATIVE_MODEL, e.g. ollama/...), then the normal chain.
                yield "narrative_llm", llm_provider.narrative_llm
                yield "llm", crewai_llm
                yield "secondary_llm", fallback_crewai_llm
                for extra in llm_provider.extra_fallback_llms():
                    yield "secondary_llm", (lambda built=extra: built)

            response = None
            last_error: Exception | None = None
            for source, factory in candidates():
                started = time.perf_counter()
                try:
                    llm = factory()
                    if llm is None:
                        continue
                    response = llm.call(prompt)
                    break
                except Exception as exc:
                    last_error = exc
            if response is None:
                raise last_error or RuntimeError("no LLM provider configured")
            try:
                metrics = llm.get_token_usage_summary()
                usage.record(
                    stage=span_name.removeprefix("llm.reasoning."),
                    agent="Orchestrator (CrewAI Flow)",
                    framework="crewai",
                    provider=str(getattr(llm, "provider", "") or provider),
                    model=str(getattr(llm, "model", "") or model),
                    prompt_tokens=metrics.prompt_tokens,
                    completion_tokens=metrics.completion_tokens,
                    total_tokens=metrics.total_tokens,
                    requests=metrics.successful_requests,
                    latency_seconds=round(time.perf_counter() - started, 3),
                    source=source,
                )
            except Exception:
                pass  # usage accounting must never break reasoning
            rendered = str(response).strip()
            if not rendered:
                raise ValueError("LLM returned an empty response")
            update_span_io(
                span,
                output={"reasoning": rendered, "fallback_used": False, "source": source},
            )
            return rendered
        except Exception as exc:
            update_span_io(
                span,
                output={
                    "reasoning": fallback,
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
            return fallback


def generate_data_prep_reasoning(
    profile_summary: dict[str, Any],
    target_col: str,
    goal: str,
) -> str:
    """Explain row-free profile findings and preparation choices."""

    prompt = (
        "You are a data scientist. Given this CSV profile (schema, null counts, "
        "row count, leakage candidates), explain in 3 bullet points what data "
        "quality issues exist and how to handle them. "
        f"Profile: {profile_summary}. Target: {target_col}. Goal: {goal}. "
        "Be specific and concise. No code."
    )
    fallback = (
        "- The schema and null counts were validated before training.\n"
        "- Leakage candidates were excluded from model features.\n"
        "- Missing values are handled with training-derived imputations before the split artifacts are used."
    )
    return _call_reasoner(
        span_name="llm.reasoning.data_prep",
        prompt=prompt,
        safe_input={
            "profile_summary": profile_summary,
            "target_column": target_col,
            "prediction_goal": goal,
        },
        fallback=fallback,
    )


def generate_model_selection_reasoning(
    candidates: list[dict[str, Any]],
    selected: str,
    metric: str,
    value: float,
) -> str:
    """Explain why the deterministic candidate ranking selected its winner."""

    prompt = (
        f"You are an ML engineer. Three models were trained: {candidates}. "
        f"The best model was {selected} with {metric}={value:.4f}. "
        "Explain in 3 bullet points WHY this model won and what trade-offs were made. "
        "Be specific about speed vs accuracy trade-offs."
    )
    fallback = (
        f"- {selected} won the deterministic ranking on {metric} ({value:.4f}).\n"
        "- Candidate training time was retained alongside quality metrics for comparison.\n"
        "- The selected model preserves the measured quality advantage while MLflow keeps the alternatives auditable."
    )
    return _call_reasoner(
        span_name="llm.reasoning.model_selection",
        prompt=prompt,
        safe_input={
            "candidates": {
                f"candidate_{index + 1}": candidate
                for index, candidate in enumerate(candidates)
            },
            "selected_model": selected,
            "metric_name": metric,
            "metric_value": value,
        },
        fallback=fallback,
    )


def generate_evaluation_reasoning(
    metrics: dict[str, float],
    passed: bool,
    threshold: float,
    feedback: str | None,
) -> str:
    """Explain the held-out gate and next action."""

    feedback_text = f"Feedback from previous attempt: {feedback}" if feedback else ""
    improvement_target = "in the next retry" if not passed else "before deployment"
    prompt = (
        f"You are an ML evaluator. Evaluation results: {metrics}. "
        f"Threshold: {threshold}. Passed: {passed}. {feedback_text} "
        "Explain in 3 bullet points: what these metrics mean, whether the model "
        f"generalizes well, and what should be improved {improvement_target}."
    )
    gate = "passed" if passed else "did not pass"
    fallback = (
        f"- The held-out metrics {gate} the configured acceptance gate of {threshold:.4f}.\n"
        "- The result reflects performance on data excluded from model fitting.\n"
        f"- Review residual or class-level errors {improvement_target}."
    )
    return _call_reasoner(
        span_name="llm.reasoning.evaluation",
        prompt=prompt,
        safe_input={
            "metrics": metrics,
            "passed": passed,
            "threshold": threshold,
            "feedback": feedback,
        },
        fallback=fallback,
    )


def generate_pipeline_summary(
    status: str,
    model_name: str,
    metric_name: str,
    metric_value: float,
    retry_count: int,
    runtime_s: float,
) -> str:
    """Summarize a completed pipeline run for a technical audience."""

    # The summary is written before deployment and reporting finish, so the
    # runtime is kept out of the text (the dashboard shows the real total).
    prompt = (
        "You are an AI system. Summarize this ML pipeline run in 2 sentences for "
        f"a technical audience. Status: {status}. Model: {model_name}. "
        f"{metric_name}: {metric_value:.4f}. Retries used: {retry_count}. "
        "Do not mention runtime or duration."
    )
    fallback = (
        f"The pipeline finished with status {status} using {model_name}; "
        f"held-out {metric_name} was {metric_value:.4f}. "
        f"It used {retry_count} retries."
    )
    return _call_reasoner(
        span_name="llm.reasoning.pipeline_summary",
        prompt=prompt,
        safe_input={
            "status": status,
            "model_name": model_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "retry_count": retry_count,
            "runtime_seconds": runtime_s,
        },
        fallback=fallback,
    )
