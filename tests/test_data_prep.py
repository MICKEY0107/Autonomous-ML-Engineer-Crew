"""Integration tests for the CrewAI data-preparation stage."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import UUID

import pandas as pd
import pytest

from agents.data_prep import (
    DataPrepError,
    PrepareDatasetTool,
    build_data_prep_agent,
    prepare_dataset,
)


RUN_ID = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")


def path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(url2pathname(unquote(parsed.path.lstrip("/"))))


def test_prepare_dataset_profiles_cleans_splits_and_writes_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "record_id,age,city,churned\n"
        "C-1,24,Delhi,0\n"
        "C-2,35,Mumbai,1\n"
        "C-3,,Delhi,0\n"
        "C-4,48,,1\n"
        "C-5,52,Jaipur,0\n"
        "C-6,31,Mumbai,1\n"
        "C-7,29,Delhi,0\n"
        "C-8,44,Jaipur,1\n"
        "C-9,39,Mumbai,0\n"
        "C-10,50,Delhi,\n"
        "C-11,41,Delhi,1\n"
        "C-12,28,Mumbai,0\n"
        "C-13,46,Jaipur,1\n",
        encoding="utf-8",
    )

    result = prepare_dataset(
        csv_path,
        target_column="churned",
        run_id=RUN_ID,
        task_type="classification",
        test_size=0.25,
        artifact_dir=tmp_path / "artifacts",
    )

    train_path = path_from_file_uri(result.train_artifact_uri)
    test_path = path_from_file_uri(result.test_artifact_uri)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    assert result.row_count == 13
    assert result.train_row_count + result.test_row_count == 12
    assert result.feature_columns == ["age", "city"]
    assert list(train.columns) == ["age", "city", "churned"]
    assert list(test.columns) == ["age", "city", "churned"]
    assert not train.isna().any().any()
    assert not test.isna().any().any()
    assert any("record_id" in finding for finding in result.quality_findings)
    assert any("missing target" in finding for finding in result.quality_findings)


def test_prepare_dataset_rejects_unknown_target(tmp_path) -> None:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "feature,label\n"
        "1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n"
        "7,0\n8,1\n9,0\n10,1\n11,0\n",
        encoding="utf-8",
    )

    with pytest.raises(DataPrepError, match="target_column"):
        prepare_dataset(
            csv_path,
            target_column="missing",
            run_id=RUN_ID,
            task_type="classification",
            artifact_dir=tmp_path / "artifacts",
        )


def test_crewai_tool_returns_validated_result_json(tmp_path) -> None:
    csv_path = tmp_path / "regression.csv"
    csv_path.write_text(
        "feature,target\n"
        "1.2,10\n3.4,20\n2.1,15\n5.6,30\n4.8,25\n6.1,35\n"
        "7.4,40\n8.2,45\n9.8,50\n10.5,55\n11.7,60\n12.3,65\n",
        encoding="utf-8",
    )
    tool = PrepareDatasetTool(artifact_dir=str(tmp_path / "artifacts"))

    payload = tool._run(
        csv_path=str(csv_path),
        target_column="target",
        run_id=RUN_ID,
        task_type="regression",
        test_size=0.2,
    )

    assert '"dataset_hash"' in payload
    assert '"train_artifact_uri"' in payload

    agent = build_data_prep_agent(artifact_dir=tmp_path / "artifacts")
    assert agent.role == "Data Preparation Engineer"
    assert [registered_tool.name for registered_tool in agent.tools] == ["prepare_dataset"]
