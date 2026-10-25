from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import orchestrator_api


def _reset_api_state() -> None:
    with orchestrator_api._lock:
        orchestrator_api._state.clear()
        orchestrator_api._state.update(
            {
                "status": "idle",
                "current_stage": None,
                "retry_count": 0,
                "stages_completed": orchestrator_api._fresh_stages(),
                "final_result": None,
                "trace_data": None,
                "error": None,
                "started_at": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def test_dashboard_status_contract_matches_frontend() -> None:
    _reset_api_state()
    client = TestClient(orchestrator_api.app)

    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/api/pipeline/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "idle"
    assert payload["current_stage"] is None
    assert payload["retry_count"] == 0
    assert [stage["name"] for stage in payload["stages_completed"]] == list(
        orchestrator_api.STAGE_NAMES
    )
    assert payload["stages"] == payload["stages_completed"]
    assert all(stage["reasoning"] == {} for stage in payload["stages_completed"])
    assert all("_started_at" not in stage for stage in payload["stages_completed"])


def test_stage_reasoning_is_returned_as_typed_row_free_metadata() -> None:
    _reset_api_state()
    orchestrator_api._set_stage(
        "Model-Selection Agent",
        "success",
        reasoning={
            "task_type": "regression",
            "candidates": [
                {"name": "ridge", "metric_name": "rmse", "metric_value": 1.25}
            ],
            "selection_reason": "ridge produced the lowest rmse",
        },
    )

    payload = TestClient(orchestrator_api.app).get("/api/pipeline/status").json()
    stage = next(
        item for item in payload["stages"] if item["name"] == "Model-Selection Agent"
    )
    assert stage["reasoning"]["task_type"] == "regression"
    assert stage["reasoning"]["candidates"][0]["metric_value"] == 1.25
    serialized = str(stage["reasoning"]).lower()
    assert "raw_rows" not in serialized
    assert "train_data" not in serialized
    assert "test_data" not in serialized


def test_dashboard_root_serves_compiled_frontend() -> None:
    client = TestClient(orchestrator_api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Autonomous ML Engineer Crew" in response.text
    assert client.get("/favicon.svg").status_code == 200


def test_dashboard_run_accepts_frontend_multipart_payload(monkeypatch) -> None:
    _reset_api_state()

    def fake_run_job(**_: object) -> None:
        with orchestrator_api._lock:
            orchestrator_api._state["status"] = "success"
            orchestrator_api._state["current_stage"] = "Terminal Result"
            orchestrator_api._state["final_result"] = {
                "status": "SUCCESS",
                "run_id": "00000000-0000-4000-8000-000000000001",
            }
            orchestrator_api._touch()

    monkeypatch.setattr(orchestrator_api, "_run_job", fake_run_job)
    client = TestClient(orchestrator_api.app)
    csv_bytes = b"feature,target\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,16\n9,18\n10,20\n11,22\n"

    response = client.post(
        "/api/pipeline/run",
        files={"csv_file": ("sample.csv", csv_bytes, "text/csv")},
        data={
            "target_column": "target",
            "prediction_goal": "predict target",
            "force_fail_eval": "false",
        },
    )

    assert response.status_code == 202
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = client.get("/api/pipeline/status").json()
        if payload["status"] == "success":
            break
        time.sleep(0.01)
    assert payload["status"] == "success"
    assert payload["final_result"]["status"] == "SUCCESS"


def test_running_stage_duration_advances_without_leaking_internal_clock() -> None:
    _reset_api_state()
    orchestrator_api._set_stage("Data-Prep Agent", "running")

    first = orchestrator_api._snapshot()
    time.sleep(0.02)
    second = orchestrator_api._snapshot()

    first_stage = next(
        stage for stage in first["stages_completed"] if stage["name"] == "Data-Prep Agent"
    )
    second_stage = next(
        stage for stage in second["stages_completed"] if stage["name"] == "Data-Prep Agent"
    )
    assert second_stage["duration_ms"] >= first_stage["duration_ms"]
    assert second_stage["duration_ms"] > 0
    assert "_started_at" not in second_stage
