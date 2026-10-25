"""Tests for the dataset-profile MCP tool."""

from __future__ import annotations

import hashlib

import pytest

from tools.profile_dataset import ProfileDatasetError, ProfileResult, profile_dataset


def test_profile_dataset_returns_validated_schema_nulls_and_hash(tmp_path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "age,city,churned\n34,Delhi,0\n,Jaipur,1\n52,,0\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(csv_path))

    assert isinstance(result, ProfileResult)
    assert result.schema["age"] == "float64"
    assert result.schema["city"] in {"object", "str", "string"}
    assert result.schema["churned"] == "int64"
    assert result.null_counts == {"age": 1, "city": 1, "churned": 0}
    assert result.dataset_hash == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert "churned" not in result.leakage_candidates


def test_profile_dataset_flags_name_identifier_and_duplicate_leakage(tmp_path) -> None:
    csv_path = tmp_path / "leaky.csv"
    csv_path.write_text(
        "customer_id,income,target_label,target_copy\n"
        "C-1,40000,0,0\n"
        "C-2,50000,1,1\n"
        "C-3,60000,0,0\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(csv_path))

    assert result.leakage_candidates == [
        "customer_id",
        "target_label",
        "target_copy",
    ]
    assert "income" not in result.leakage_candidates


def test_profile_dataset_rejects_malformed_csv(tmp_path) -> None:
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("feature,target\n1,yes\n2,no,unexpected\n", encoding="utf-8")

    with pytest.raises(ProfileDatasetError, match="empty or malformed"):
        profile_dataset(str(csv_path))
