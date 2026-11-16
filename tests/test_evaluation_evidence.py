"""Evaluation evidence is portable, typed, and row-safe."""

from __future__ import annotations

from pathlib import Path

from orchestrator import evaluation_evidence


RUN_ID = "10000000-0000-4000-8000-000000000001"


def _payload() -> dict:
    return {
        "final_result": {
            "status": "SUCCESS",
            "run_id": RUN_ID,
            "metric_name": "f1_weighted",
            "metric_value": 0.91,
            "threshold": 0.8,
            "runtime_ms": 1234,
            "retry_count": 0,
            "token_count": 42,
            "estimated_cost_usd": 0.00012,
            "trace_url": "https://example.test/trace",
            "mlflow_run_id": "mlflow-1",
            "mlflow_url": "https://example.test/mlflow",
            "usage_report": {"input_tokens": 20, "output_tokens": 22, "total_tokens": 42},
        },
        "run_input": {"dataset_name": "safe.csv", "target_column": "target"},
        "insights": {"dataset": {"rows": 100, "columns": 3, "dataset_hash": "a" * 64, "task_type": "classification"}},
        "stages_completed": [
            {"name": "Data-Prep Agent", "status": "success", "duration_ms": 100, "detail": "ok", "reasoning": {"raw_rows": [{"secret": "never export"}]}},
            {"name": "A2A: message/send", "status": "success", "duration_ms": 50, "detail": "ok", "reasoning": {}},
            {"name": "MCP: profile_dataset", "status": "success", "duration_ms": 10, "detail": "ok", "reasoning": {}},
            {"name": "Evaluation Agent", "status": "success", "duration_ms": 20, "detail": "ok", "reasoning": {"metrics": [{"name": "f1_weighted", "value": 0.91}]}},
        ],
        "callback_events": [{"name": "flow.data_prep", "status": "success", "detail": "ok", "metadata": {}}],
        "a2a": {"exchanges": [{"method": "SendMessage", "task_state": "TASK_STATE_COMPLETED", "latency_ms": 50, "agent": "model_selection_agent", "task_id": "task-1", "protocol_version": "0.3"}]},
    }


def test_build_evidence_keeps_ml_quality_separate_and_redacts_rows() -> None:
    evidence = evaluation_evidence.build_evidence(_payload())

    assert evidence.model_quality == {"f1_weighted": 0.91}
    assert evidence.primary_metric == "f1_weighted"
    assert any(event.category == "a2a" for event in evidence.events)
    assert any(event.category == "mcp" for event in evidence.events)
    assert "never export" not in evidence.model_dump_json()


def test_local_evidence_store_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_evidence, "LOCAL_EVIDENCE_DIR", tmp_path)
    monkeypatch.delenv("EVALUATION_EVIDENCE_BUCKET", raising=False)
    evidence = evaluation_evidence.build_evidence(_payload())

    location = evaluation_evidence.save_evidence(evidence)

    assert location["backend"] == "local"
    assert evaluation_evidence.get_evidence(RUN_ID).run_id == evidence.run_id
    assert [item.run_id for item in evaluation_evidence.list_evidence()] == [evidence.run_id]
