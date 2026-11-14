"""Pytest integration for the ADK Model Selection Agent evaluation set.

Runs the 6 evaluation cases from tests/adk_eval/evalset.json using pytest,
verifying that the Google ADK agent satisfies contract, protocol, and ML
quality criteria.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from tests.adk_eval.run_eval import (
    _run_classification_success,
    _run_regression_success,
    _run_retry_with_feedback,
    _run_bad_input_rejected,
    _run_injection_blocked,
    _run_classification_all_metrics,
)


@pytest.fixture
def eval_tmp(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    db = (tmp_path / "mlflow_eval.db").resolve()
    return f"sqlite:///{db.as_posix()}"


def test_adk_eval_classification_success(eval_tmp: Path, tracking_uri: str) -> None:
    """Test 1: Valid classification request produces 3 candidates, selected model, F1 metric."""
    result = _run_classification_success(eval_tmp, tracking_uri)
    for check_name, passed in result["checks"].items():
        assert passed, f"Check failed: {check_name}"

def test_adk_eval_regression_success(eval_tmp: Path, tracking_uri: str) -> None:
    """Test 2: Valid regression request produces 3 candidates, selected model, RMSE metric."""
    result = _run_regression_success(eval_tmp, tracking_uri)
    for check_name, passed in result["checks"].items():
        assert passed, f"Check failed: {check_name}"


def test_adk_eval_retry_with_feedback(eval_tmp: Path, tracking_uri: str) -> None:
    """Test 3: Retry attempt 1 with feedback completes and records retry_attempt=1."""
    result = _run_retry_with_feedback(eval_tmp, tracking_uri)
    for check_name, passed in result["checks"].items():
        assert passed, f"Check failed: {check_name}"


def test_adk_eval_bad_input_rejected() -> None:
    """Test 4: Malformed request is rejected before executing training."""
    result = _run_bad_input_rejected()
    for check_name, passed in result["checks"].items():
        assert passed, f"Check failed: {check_name}"


def test_adk_eval_injection_blocked() -> None:
    """Test 5: Injection goal is rejected by the ADK callback before training."""
    result = _run_injection_blocked()
    for check_name, passed in result["checks"].items():
        assert passed, f"Check failed: {check_name}"


def test_adk_eval_classification_all_metrics(eval_tmp: Path, tracking_uri: str) -> None:
    """Test 6: Full classification candidate logging with all metrics."""
    result = _run_classification_all_metrics(eval_tmp, tracking_uri)
    for check_name, passed in result["checks"].items():
        assert passed, f"Check failed: {check_name}"
