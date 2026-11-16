"""Regression coverage for the Cloud Run Job's sequential-run handoff."""

from __future__ import annotations

from scripts import run_cloud_evaluation as runner


def test_start_pipeline_retries_transient_active_worker_conflict(monkeypatch) -> None:
    starts = 0

    def fake_request(path: str, **_: object) -> tuple[int, dict]:
        nonlocal starts
        if path == "/api/pipeline/run":
            starts += 1
            return (409, {"detail": "A pipeline run is already active"}) if starts == 1 else (202, {})
        if path == "/api/pipeline/status":
            return 200, {"status": "success"}
        if path == "/api/pipeline/result":
            return 200, {"final_result": {"status": "SUCCESS"}}
        raise AssertionError(path)

    monkeypatch.setattr(runner, "_request", fake_request)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    result, elapsed = runner._start_pipeline(force_fail=True)

    assert starts == 2
    assert result["final_result"]["status"] == "SUCCESS"
    assert elapsed >= 0
