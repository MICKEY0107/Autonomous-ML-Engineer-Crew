"""Reusable guardrails for untrusted agent and pipeline input."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel


MAX_PREDICTION_GOAL_CHARS = 2_000
PROMPT_INJECTION_KEYWORDS = frozenset(
    {
        "bypass safety",
        "developer message",
        "disregard previous",
        "execute shell",
        "forget previous",
        "ignore all instructions",
        "ignore previous instructions",
        "jailbreak",
        "override system",
        "prompt injection",
        "reveal secrets",
        "run command",
        "system prompt",
    }
)
PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override)\b.{0,80}"
        r"\b(?:instruction|message|policy|prompt|rule)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(?m)^\s*(?:assistant|developer|system)\s*:", re.IGNORECASE),
    re.compile(r"<\|\s*(?:assistant|developer|system|user)\s*\|>", re.IGNORECASE),
    re.compile(r"\b(?:show|reveal|print|leak)\b.{0,50}\bsystem prompt\b", re.IGNORECASE),
)
TOXIC_TERMS = frozenset(
    {
        "asshole",
        "bastard",
        "bitch",
        "bullshit",
        "dickhead",
        "fuck",
        "fucking",
        "motherfucker",
        "retard",
        "shit",
        "slut",
        "whore",
    }
)
TOXIC_PATTERNS = (
    re.compile(
        r"\b(?:kill|hurt|murder|rape|shoot|stab|attack)\s+"
        r"(?:you|him|her|them|people|everyone|myself|yourself)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i|we)\s+(?:hate|despise)\s+(?:you|all|them|those)\b", re.IGNORECASE),
    re.compile(r"\b(?:go\s+die|kys)\b", re.IGNORECASE),
    re.compile(r"\b(?:stupid|dumb|useless)\s+(?:bot|ai|idiot|moron)\b", re.IGNORECASE),
)
_TOXIC_WORD = re.compile(r"[a-z]+")
_FORBIDDEN_ROW_KEYS = frozenset(
    {
        "dataframe",
        "dataset_rows",
        "raw_data",
        "raw_rows",
        "records",
        "rows",
        "samples",
        "test_data",
        "training_data",
    }
)
_SAFE_PROMPT_FIELDS = frozenset(
    {
        "artifact_uris",
        "dataset_uri",
        "evaluation_metrics",
        "prediction_goal",
        "retry_feedback",
        "run_id",
        "schema",
        "target_column",
        "task_type",
        "training_metrics",
    }
)


class GuardrailViolation(ValueError):
    """Raised when untrusted input fails a pipeline boundary guardrail."""


def find_toxic_language(text: str) -> list[str]:
    """Return the toxicity categories found in ``text`` (empty when clean)."""

    normalized = " ".join(str(text).casefold().split())
    found: list[str] = []
    if any(word in TOXIC_TERMS for word in _TOXIC_WORD.findall(normalized)):
        found.append("profanity_or_abuse")
    if any(pattern.search(normalized) for pattern in TOXIC_PATTERNS):
        found.append("threat_or_harassment")
    return found


def validate_prediction_goal(prediction_goal: str) -> str:
    """Reject empty, oversized, injected, toxic, or abusive prediction goals."""

    goal = str(prediction_goal).strip()
    if not goal:
        raise GuardrailViolation("prediction_goal must not be blank")
    if len(goal) > MAX_PREDICTION_GOAL_CHARS:
        raise GuardrailViolation(
            f"prediction_goal must not exceed {MAX_PREDICTION_GOAL_CHARS} characters"
        )

    normalized = " ".join(goal.casefold().split())
    matched_keyword = next(
        (keyword for keyword in PROMPT_INJECTION_KEYWORDS if keyword in normalized),
        None,
    )
    if matched_keyword or any(pattern.search(goal) for pattern in PROMPT_INJECTION_PATTERNS):
        raise GuardrailViolation(
            "prediction_goal contains prompt-injection-like instructions"
        )
    toxic = find_toxic_language(goal)
    if toxic:
        raise GuardrailViolation(
            "prediction_goal contains toxic or abusive language "
            f"({', '.join(toxic)})"
        )
    return goal


def assert_llm_prompt_safe(payload: BaseModel | Mapping[str, Any]) -> None:
    """Fail if a prompt context contains raw rows or non-metadata fields."""

    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    if not isinstance(data, Mapping):
        raise GuardrailViolation("LLM prompt context must be a metadata object")
    unexpected = set(data) - _SAFE_PROMPT_FIELDS
    if unexpected:
        raise GuardrailViolation(
            "LLM prompt context contains non-metadata fields: "
            + ", ".join(sorted(str(key) for key in unexpected))
        )

    def walk(value: Any) -> None:
        if isinstance(value, pd.DataFrame):
            raise GuardrailViolation("pandas DataFrames cannot be passed to an LLM")
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = re.sub(
                    r"[^a-z0-9]+", "_", str(key).casefold()
                ).strip("_")
                if normalized_key in _FORBIDDEN_ROW_KEYS:
                    raise GuardrailViolation(
                        f"raw dataset rows are forbidden in LLM prompt field '{key}'"
                    )
                walk(child)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if any(isinstance(item, Mapping) for item in value):
                raise GuardrailViolation(
                    "lists of records cannot be passed to an LLM prompt"
                )
            for item in value:
                walk(item)

    walk(data)
