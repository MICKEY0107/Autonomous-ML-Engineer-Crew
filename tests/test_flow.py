"""State-machine tests for the CrewAI orchestration Flow."""

from __future__ import annotations

from uuid import UUID

import pytest

from contracts.models import (
    DataPrepResult,
    EvaluationResult,
    ModelSelectionResult,
    PipelineState,
)
from orchestrator.flow import (
    AutonomousMLEngineerFlow,
    FlowCallbackEvent,
    FlowCallbacks,
)


RUN_ID = UUID("40000000-0000-4000-8000-000000000001")


def _state() -> PipelineState:
    return PipelineState(
        run_id=RUN_ID,
        dataset_uri="file:///tmp/flow-input.csv",
        task_type="classification",
        target_column="target",
        prediction_goal="Predict the synthetic target.",
        trace_id="abcdef0123456789",
    )


def _script_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pass_on_attempt: int,
    fail_prep: bool = False,
) -> list[str | None]:
    sent_feedback: list[str | None] = []

    def prepare(flow: AutonomousMLEngineerFlow) -> DataPrepResult:
        if fail_prep:
            raise ValueError("synthetic input validation failure")
        return DataPrepResult(
            run_id=flow.state.run_id,
            dataset_hash="e" * 64,
            schema_summary={"feature": "float64", "target": "int64"},
            quality_findings=["No data-quality issues were detected."],
            target_column="target",
            feature_columns=["feature"],
            row_count=20,
            train_row_count=16,
            test_row_count=4,
            train_artifact_uri="file:///tmp/flow-train.csv",
            test_artifact_uri="file:///tmp/flow-test.csv",
        )

    def select(
        flow: AutonomousMLEngineerFlow,
        retry_feedback: str | None = None,
    ) -> ModelSelectionResult:
        sent_feedback.append(retry_feedback)
        attempt = flow.state.retry_count
        run_id = f"candidate-{attempt}"
        return ModelSelectionResult(
            run_id=flow.state.run_id,
            candidate_run_ids=[run_id],
            best_run_id=run_id,
            best_model_uri=f"models:/scripted/{attempt + 1}",
            best_parameters={"model_type": "random_forest"},
            training_metrics={"f1_weighted": 0.7 + attempt / 10},
        )

    def evaluate(flow: AutonomousMLEngineerFlow) -> EvaluationResult:
        passed = flow.state.retry_count >= pass_on_attempt
        observed = 0.90 if passed else 0.40
        return EvaluationResult(
            run_id=flow.state.run_id,
            model_uri=flow.state.model_selection_result.best_model_uri,
            held_out_metrics={"f1_weighted": observed},
            primary_metric="f1_weighted",
            acceptance_threshold=0.80,
            metric_direction="maximize",
            threshold_rationale="Scripted weighted-F1 acceptance gate.",
            passed=passed,
            retry_feedback=(
                None if passed else "Increase minority-class recall in the next attempt."
            ),
        )

    monkeypatch.setattr(AutonomousMLEngineerFlow, "_prepare_data", prepare)
    monkeypatch.setattr(AutonomousMLEngineerFlow, "_post_model_selection", select)
    monkeypatch.setattr(AutonomousMLEngineerFlow, "_evaluate_selection", evaluate)
    return sent_feedback


def _recording_callbacks(events: list[FlowCallbackEvent]) -> FlowCallbacks:
    def record(event: FlowCallbackEvent) -> None:
        events.append(event)

    return FlowCallbacks(
        on_stage_start=record,
        on_stage_complete=record,
        on_stage_error=record,
        on_retry=record,
        on_validation_failure=record,
    )


def test_flow_reaches_success_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[FlowCallbackEvent] = []
    _script_flow(monkeypatch, pass_on_attempt=0)
    flow = AutonomousMLEngineerFlow(
        initial_state=_state(),
        callbacks=_recording_callbacks(events),
        suppress_flow_events=True,
    )

    flow.kickoff()

    assert flow.state.status == "SUCCESS"
    assert flow.state.retry_count == 0
    assert flow.state.evaluation_result.passed is True
    assert [event.name for event in events].count("on_stage_start") == 3
    assert [event.name for event in events].count("on_stage_complete") == 3


def test_flow_sends_feedback_and_succeeds_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[FlowCallbackEvent] = []
    sent_feedback = _script_flow(monkeypatch, pass_on_attempt=1)
    flow = AutonomousMLEngineerFlow(
        initial_state=_state(),
        callbacks=_recording_callbacks(events),
        suppress_flow_events=True,
    )

    flow.kickoff()

    assert flow.state.status == "SUCCESS"
    assert flow.state.retry_count == 1
    assert sent_feedback == [None, "Increase minority-class recall in the next attempt."]
    retry_events = [event for event in events if event.name == "on_retry"]
    assert len(retry_events) == 1
    assert retry_events[0].retry_count == 1


def test_flow_stops_after_two_failed_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[FlowCallbackEvent] = []
    _script_flow(monkeypatch, pass_on_attempt=99)
    flow = AutonomousMLEngineerFlow(
        initial_state=_state(),
        callbacks=_recording_callbacks(events),
        suppress_flow_events=True,
    )

    flow.kickoff()

    assert flow.state.status == "FAILED_EVAL_EXHAUSTED"
    assert flow.state.retry_count == 2
    assert flow.state.errors
    assert len([event for event in events if event.name == "on_retry"]) == 2


def test_flow_invalid_input_fires_validation_and_error_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[FlowCallbackEvent] = []
    _script_flow(monkeypatch, pass_on_attempt=0, fail_prep=True)
    flow = AutonomousMLEngineerFlow(
        initial_state=_state(),
        callbacks=_recording_callbacks(events),
        suppress_flow_events=True,
    )

    flow.kickoff()

    assert flow.state.status == "FAILED_INPUT"
    assert flow.state.errors
    names = [event.name for event in events]
    assert "on_validation_failure" in names
    assert "on_stage_error" in names


def test_all_nine_callback_invocations_are_correlated_by_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[FlowCallbackEvent] = []
    _script_flow(monkeypatch, pass_on_attempt=0)
    successful = AutonomousMLEngineerFlow(
        initial_state=_state(),
        callbacks=_recording_callbacks(events),
        suppress_flow_events=True,
    )
    successful.kickoff()

    _script_flow(monkeypatch, pass_on_attempt=0, fail_prep=True)
    invalid = AutonomousMLEngineerFlow(
        initial_state=_state(),
        callbacks=_recording_callbacks(events),
        suppress_flow_events=True,
    )
    invalid.kickoff()

    assert len(events) == 9
    assert {event.name for event in events} == {
        "on_stage_start",
        "on_stage_complete",
        "on_stage_error",
        "on_validation_failure",
    }
    assert {event.run_id for event in events} == {RUN_ID}
