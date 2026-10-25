"""Trace hierarchy, propagation, and redaction tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pandas as pd
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from agents.data_prep import prepare_dataset
from agents.model_selection_adk import _experiment_call
from contracts.models import (
    DataPrepResult,
    EvaluationResult,
    ModelSelectionResult,
    PipelineState,
)
from orchestrator.flow import AutonomousMLEngineerFlow
from orchestrator.telemetry import (
    REDACTED,
    get_tracer_provider,
    langfuse_trace_url,
    pipeline_trace,
    redact_payload,
    redact_text,
    telemetry_attributes,
    traced_span,
)


RUN_ID = UUID("50000000-0000-4000-8000-000000000001")
_EXPORTER = InMemorySpanExporter()
get_tracer_provider().add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture(autouse=True)
def clear_spans() -> None:
    _EXPORTER.clear()


def test_langfuse_trace_url_uses_pipeline_run_id_project_route() -> None:
    assert langfuse_trace_url(RUN_ID) == (
        "https://cloud.langfuse.com/project/cmvjt4kam04c9ad0jy0a92aum/"
        "traces/50000000000040008000000000000001"
    )


def _state() -> PipelineState:
    return PipelineState(
        run_id=RUN_ID,
        dataset_uri="file:///tmp/trace-input.csv",
        task_type="classification",
        target_column="target",
        prediction_goal="Predict the synthetic target.",
        trace_id="abcdef0123456789",
    )


def _data_prep_result() -> DataPrepResult:
    return DataPrepResult(
        run_id=RUN_ID,
        dataset_hash="e" * 64,
        schema_summary={"feature": "float64", "target": "int64"},
        target_column="target",
        feature_columns=["feature"],
        row_count=20,
        train_row_count=16,
        test_row_count=4,
        train_artifact_uri="file:///tmp/trace-train.csv",
        test_artifact_uri="file:///tmp/trace-test.csv",
    )


def test_flow_uses_run_id_trace_with_one_span_per_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepare(_: AutonomousMLEngineerFlow) -> DataPrepResult:
        return _data_prep_result()

    def select(
        _: AutonomousMLEngineerFlow,
        retry_feedback: str | None = None,
    ) -> ModelSelectionResult:
        return ModelSelectionResult(
            run_id=RUN_ID,
            candidate_run_ids=["candidate-1"],
            best_run_id="candidate-1",
            best_model_uri="models:/trace-model/1",
            best_parameters={"model_type": "random_forest"},
            training_metrics={"f1_weighted": 0.91},
        )

    def evaluate(flow: AutonomousMLEngineerFlow) -> EvaluationResult:
        return EvaluationResult(
            run_id=RUN_ID,
            model_uri=flow.state.model_selection_result.best_model_uri,
            held_out_metrics={"f1_weighted": 0.90},
            primary_metric="f1_weighted",
            acceptance_threshold=0.80,
            metric_direction="maximize",
            threshold_rationale="Synthetic trace gate.",
            passed=True,
        )

    monkeypatch.setattr(AutonomousMLEngineerFlow, "_prepare_data", prepare)
    monkeypatch.setattr(AutonomousMLEngineerFlow, "_post_model_selection", select)
    monkeypatch.setattr(AutonomousMLEngineerFlow, "_evaluate_selection", evaluate)
    flow = AutonomousMLEngineerFlow(initial_state=_state(), suppress_flow_events=True)

    flow.kickoff()

    spans = list(_EXPORTER.get_finished_spans())
    root = next(span for span in spans if span.name == "pipeline.run")
    agents = [span for span in spans if span.name.startswith("agent.")]
    assert f"{root.context.trace_id:032x}" == RUN_ID.hex
    assert flow.state.trace_id == RUN_ID.hex
    assert {span.name for span in agents} == {
        "agent.data_prep",
        "agent.model_selection",
        "agent.evaluation",
    }
    assert all(span.parent.span_id == root.context.span_id for span in agents)

    root_input = json.loads(root.attributes["langfuse.observation.input"])
    root_output = json.loads(root.attributes["langfuse.observation.output"])
    assert set(root_input) == {"csv_path", "target_column", "prediction_goal"}
    assert root_input["target_column"] == "target"
    assert root_input["prediction_goal"] == "Predict the synthetic target."
    assert root_output == {"run_id": str(RUN_ID), "status": "SUCCESS"}

    agents_by_name = {span.name: span for span in agents}
    data_prep_input = json.loads(
        agents_by_name["agent.data_prep"].attributes[
            "langfuse.observation.input"
        ]
    )
    data_prep_output = json.loads(
        agents_by_name["agent.data_prep"].attributes[
            "langfuse.observation.output"
        ]
    )
    assert set(data_prep_input) == {"csv_path", "target_column"}
    assert set(data_prep_output) == {"row_count", "test_path", "train_path"}
    assert data_prep_output["row_count"] == 20
    assert Path(data_prep_output["test_path"]).name == "trace-test.csv"
    assert Path(data_prep_output["train_path"]).name == "trace-train.csv"

    selection_input = json.loads(
        agents_by_name["agent.model_selection"].attributes[
            "langfuse.observation.input"
        ]
    )
    selection_output = json.loads(
        agents_by_name["agent.model_selection"].attributes[
            "langfuse.observation.output"
        ]
    )
    assert set(selection_input) == {"task_type", "retry_count"}
    assert selection_output == {
        "best_model_name": "random_forest",
        "metric_value": 0.91,
    }

    evaluation_input = json.loads(
        agents_by_name["agent.evaluation"].attributes[
            "langfuse.observation.input"
        ]
    )
    evaluation_output = json.loads(
        agents_by_name["agent.evaluation"].attributes[
            "langfuse.observation.output"
        ]
    )
    assert evaluation_input == {
        "metric_value": 0.90,
        "threshold": 0.8,
    }
    assert evaluation_output["passed"] is True
    assert evaluation_output["feedback"]


def test_profile_mcp_span_is_child_of_data_prep_agent(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    pd.DataFrame(
        {
            "feature": [value / 10 for value in range(12)],
            "target": [value % 2 for value in range(12)],
        }
    ).to_csv(csv_path, index=False)

    with pipeline_trace(RUN_ID):
        with traced_span("agent.data_prep"):
            prepare_dataset(
                csv_path,
                target_column="target",
                run_id=RUN_ID,
                task_type="classification",
                artifact_dir=tmp_path / "artifacts",
            )

    spans = {span.name: span for span in _EXPORTER.get_finished_spans()}
    assert spans["mcp.profile_dataset"].parent.span_id == spans[
        "agent.data_prep"
    ].context.span_id
    assert spans["mcp.profile_dataset"].attributes["mcp.result.validated"] is True
    profile_input = json.loads(
        spans["mcp.profile_dataset"].attributes["langfuse.observation.input"]
    )
    profile_output = json.loads(
        spans["mcp.profile_dataset"].attributes["langfuse.observation.output"]
    )
    assert profile_input == {"csv_path": str(csv_path.resolve())}
    assert set(profile_output) == {"sha256", "null_count", "feature_count"}
    assert len(profile_output["sha256"]) == 64
    assert profile_output["null_count"] == 0
    assert profile_output["feature_count"] == 2


def test_mlflow_mcp_span_is_child_and_redacts_arguments() -> None:
    class FakeSession:
        async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            assert name == "log_and_query_experiment"
            return SimpleNamespace(
                structuredContent={
                    "action": "create_run",
                    "experiment_id": "1",
                    "run_id": "mlflow-run-1",
                    "status": "RUNNING",
                },
                isError=False,
            )

    with pipeline_trace(RUN_ID):
        with traced_span("agent.model_selection"):
            asyncio.run(
                _experiment_call(
                    FakeSession(),
                    "sqlite:///tmp/trace.db",
                    "create_run",
                    experiment_name="trace-test",
                    params={"model": "rf"},
                    raw_rows=[{"secret": "must-not-appear"}],
                )
            )

    spans = {span.name: span for span in _EXPORTER.get_finished_spans()}
    mcp_span = spans["mcp.log_and_query_experiment"]
    assert mcp_span.parent.span_id == spans["agent.model_selection"].context.span_id
    assert "must-not-appear" not in mcp_span.attributes["mcp.arguments"]
    assert REDACTED in mcp_span.attributes["mcp.arguments"]
    assert json.loads(mcp_span.attributes["langfuse.observation.input"]) == {
        "experiment_name": "trace-test",
        "params": {"model": "rf"},
    }
    assert json.loads(mcp_span.attributes["langfuse.observation.output"]) == {
        "best_metric": None,
        "run_id": "mlflow-run-1",
    }


def test_a2a_span_has_task_retry_attributes_and_propagates_traceparent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    state.data_prep_result = _data_prep_result()
    state.retry_count = 1
    flow = AutonomousMLEngineerFlow(initial_state=state, suppress_flow_events=True)
    captured_headers: dict[str, str] = {}

    def send(base_url: str, payload: dict, **kwargs: object):
        from contracts.models import ModelSelectionResult
        from orchestrator.a2a_client import A2ACallInfo

        captured_headers.update(kwargs["headers"])
        result = ModelSelectionResult.model_validate(
            {
                "run_id": str(RUN_ID),
                "candidate_run_ids": ["candidate-1"],
                "best_run_id": "candidate-1",
                "best_model_uri": "models:/trace-model/1",
                "best_parameters": {"model_type": "random_forest"},
                "training_metrics": {"f1_weighted": 0.91},
            }
        )
        info = A2ACallInfo(
            agent_name="model_selection_agent",
            task_id="task-1",
            task_state="TASK_STATE_COMPLETED",
        )
        return result, info

    monkeypatch.setattr("orchestrator.flow.send_model_selection_task", send)
    with pipeline_trace(RUN_ID):
        with traced_span("agent.model_selection"):
            flow._post_model_selection("Improve recall")

    spans = {span.name: span for span in _EXPORTER.get_finished_spans()}
    a2a = spans["a2a.message.send"]
    assert a2a.attributes["ml.task_type"] == "classification"
    assert a2a.attributes["ml.retry_count"] == 1
    assert a2a.attributes["rpc.method"] == "SendMessage"
    assert a2a.attributes["a2a.task.state"] == "TASK_STATE_COMPLETED"
    assert captured_headers["traceparent"].split("-")[1] == RUN_ID.hex
    assert json.loads(a2a.attributes["langfuse.observation.input"]) == {
        "retry_count": 1,
        "task_type": "classification",
    }
    assert json.loads(a2a.attributes["langfuse.observation.output"]) == {
        "best_model_name": "random_forest",
        "metric_value": 0.91,
    }


def test_redaction_removes_row_fields_before_logging_or_export() -> None:
    payload = {
        "schema": {"feature": "int64"},
        "raw_rows": [{"secret": "alpha"}],
        "nested": {"training_data": [{"secret": "beta"}]},
    }

    redacted = redact_payload(payload)
    attributes = telemetry_attributes({"event.payload": payload})
    message = redact_text('failure raw_rows=[{"secret":"gamma"}]')
    serialized = json.dumps(redacted)

    assert REDACTED in serialized
    assert "alpha" not in serialized
    assert "beta" not in serialized
    assert "alpha" not in attributes["event.payload"]
    assert "gamma" not in message
