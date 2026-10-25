from __future__ import annotations

import pytest

from orchestrator import llm_reasoning


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.mark.parametrize(
    ("call", "expected_prompt_text"),
    [
        (
            lambda: llm_reasoning.generate_data_prep_reasoning(
                {
                    "schema": {"feature": "float64", "target": "float64"},
                    "null_counts": {"feature": 2, "target": 0},
                    "row_count": 50,
                    "leakage_candidates": [],
                },
                "target",
                "predict target",
            ),
            "data quality issues",
        ),
        (
            lambda: llm_reasoning.generate_model_selection_reasoning(
                [{"name": "ridge", "metric": "rmse", "value": 1.2}],
                "ridge",
                "rmse",
                1.2,
            ),
            "speed vs accuracy",
        ),
        (
            lambda: llm_reasoning.generate_evaluation_reasoning(
                {"rmse": 1.2, "r2": 0.9},
                True,
                2.0,
                None,
            ),
            "generalizes well",
        ),
        (
            lambda: llm_reasoning.generate_pipeline_summary(
                "SUCCESS",
                "ridge",
                "r2",
                0.992,
                0,
                6.5,
            ),
            "technical audience",
        ),
    ],
)
def test_reasoning_helpers_call_configured_llm(
    monkeypatch: pytest.MonkeyPatch,
    call,
    expected_prompt_text: str,
) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    fake = _FakeLlm("- Evidence-based Gemini explanation")
    monkeypatch.setattr(llm_reasoning, "crewai_llm", lambda: fake)

    result = call()

    assert result == "- Evidence-based Gemini explanation"
    assert len(fake.prompts) == 1
    assert expected_prompt_text in fake.prompts[0]


def test_reasoning_failure_returns_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")

    def fail_provider():
        raise KeyError("missing provider credential")

    monkeypatch.setattr(llm_reasoning, "crewai_llm", fail_provider)
    monkeypatch.setattr(llm_reasoning, "fallback_crewai_llm", lambda: None)

    result = llm_reasoning.generate_pipeline_summary(
        "SUCCESS",
        "ridge",
        "r2",
        0.992,
        0,
        6.5,
    )

    assert "SUCCESS" in result
    assert "ridge" in result
    assert "0.9920" in result


def test_reasoning_uses_secondary_llm_before_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODE", "live")

    def fail_primary():
        raise RuntimeError("primary quota exhausted")

    alternate = _FakeLlm("- Alternate provider explanation")
    monkeypatch.setattr(llm_reasoning, "crewai_llm", fail_primary)
    monkeypatch.setattr(
        llm_reasoning,
        "fallback_crewai_llm",
        lambda: alternate,
    )

    result = llm_reasoning.generate_pipeline_summary(
        "SUCCESS",
        "ridge",
        "r2",
        0.992,
        0,
        6.5,
    )

    assert result == "- Alternate provider explanation"
    assert len(alternate.prompts) == 1
