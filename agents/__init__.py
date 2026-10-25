"""Agent definitions for the Autonomous ML Engineer Crew."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_data_prep_agent",
    "prepare_dataset",
    "run_data_prep",
    "build_evaluation_agent",
    "evaluate_model",
    "run_evaluation",
]


def __getattr__(name: str) -> Any:
    """Load CrewAI only when its data-preparation exports are requested."""

    if name in {"build_data_prep_agent", "prepare_dataset", "run_data_prep"}:
        from .data_prep import build_data_prep_agent, prepare_dataset, run_data_prep

        return {
            "build_data_prep_agent": build_data_prep_agent,
            "prepare_dataset": prepare_dataset,
            "run_data_prep": run_data_prep,
        }[name]
    if name in {"build_evaluation_agent", "evaluate_model", "run_evaluation"}:
        from .evaluation import build_evaluation_agent, evaluate_model, run_evaluation

        return {
            "build_evaluation_agent": build_evaluation_agent,
            "evaluate_model": evaluate_model,
            "run_evaluation": run_evaluation,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
