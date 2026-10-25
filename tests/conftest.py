"""Shared synthetic-data fixtures for integration tests."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
import pytest


# Unit and integration tests exercise deterministic fallback adapters. Live LLM
# behavior is verified separately through the local end-to-end demo.
os.environ["LLM_MODE"] = "offline"


@pytest.fixture
def small_csv() -> Path:
    """Create a reusable 50-row supervised-learning CSV under ``/tmp``."""

    generator = random.Random(42)
    path = Path("/tmp/test_input.csv").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "feature1": [generator.uniform(-2, 2) for _ in range(50)],
            "feature2": [generator.uniform(0, 10) for _ in range(50)],
            "feature3": [generator.uniform(20, 50) for _ in range(50)],
            "target": [index % 2 for index in range(50)],
        }
    )
    frame.to_csv(path, index=False)
    yield path
    path.unlink(missing_ok=True)
