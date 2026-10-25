"""MCP tool for profiling a CSV without exposing its raw rows.

Run this module directly to serve the tool over MCP's stdio transport::

    python -m tools.profile_dataset
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Annotated

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_CSV_BYTES = 100 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_LEAKAGE_NAME_TOKENS = frozenset(
    {
        "label",
        "outcome",
        "prediction",
        "predicted",
        "response",
        "result",
        "target",
    }
)
_IDENTIFIER_NAME_TOKENS = frozenset({"id", "identifier", "key", "uuid"})


class ProfileResult(BaseModel):
    """Validated, row-free summary returned across the MCP boundary."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )

    column_schema: dict[str, str] = Field(
        alias="schema",
        min_length=1,
        description="Mapping of CSV column names to pandas dtype names.",
    )
    null_counts: dict[str, int] = Field(
        min_length=1,
        description="Number of null values in each column.",
    )
    leakage_candidates: list[str] = Field(
        default_factory=list,
        description="Columns whose names or values merit a leakage review.",
    )
    dataset_hash: str = Field(
        description="Lowercase SHA-256 digest of the exact CSV bytes.",
    )

    @property
    def schema(self) -> dict[str, str]:
        """Expose the public contract name without shadowing a Pydantic field."""

        return self.column_schema

    @property
    def sha256(self) -> str:
        """Compatibility name for the dataset's SHA-256 digest."""

        return self.dataset_hash

    @field_validator("column_schema")
    @classmethod
    def validate_schema(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not name.strip() or not dtype.strip() for name, dtype in value.items()):
            raise ValueError("schema column names and dtypes must not be blank")
        return value

    @field_validator("null_counts")
    @classmethod
    def validate_null_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not name.strip() for name in value):
            raise ValueError("null-count column names must not be blank")
        if any(count < 0 for count in value.values()):
            raise ValueError("null counts cannot be negative")
        return value

    @field_validator("dataset_hash")
    @classmethod
    def validate_dataset_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("dataset_hash must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_matching_columns(self) -> "ProfileResult":
        if self.schema.keys() != self.null_counts.keys():
            raise ValueError("schema and null_counts must describe the same columns")
        unknown_candidates = set(self.leakage_candidates) - set(self.schema)
        if unknown_candidates:
            raise ValueError("leakage_candidates must refer to columns in schema")
        if len(set(self.leakage_candidates)) != len(self.leakage_candidates):
            raise ValueError("leakage_candidates must be unique")
        return self


class ProfileDatasetError(ValueError):
    """Safe error raised when a dataset cannot be profiled."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as csv_file:
        for chunk in iter(lambda: csv_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name_tokens(column_name: str) -> set[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name)
    return {token for token in re.split(r"[^a-z0-9]+", snake_case.lower()) if token}


def _is_sequential_index(series: pd.Series) -> bool:
    if series.isna().any() or not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.to_numpy()
    length = len(values)
    if length == 0:
        return False
    zero_based = values == range(length)
    one_based = values == range(1, length + 1)
    return bool(zero_based.all() or one_based.all())


def _find_leakage_candidates(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    row_count = len(frame)

    for column_name in frame.columns:
        series = frame[column_name]
        tokens = _name_tokens(str(column_name))
        name_suggests_leakage = bool(tokens & _LEAKAGE_NAME_TOKENS)
        is_unique_identifier = (
            row_count > 1
            and bool(tokens & _IDENTIFIER_NAME_TOKENS)
            and series.nunique(dropna=False) == row_count
        )

        if name_suggests_leakage or is_unique_identifier or _is_sequential_index(series):
            candidates.append(str(column_name))

    # Identical columns can accidentally reveal a target under another name.
    columns = [str(column) for column in frame.columns]
    for index, left_name in enumerate(columns):
        for right_name in columns[index + 1 :]:
            if frame[left_name].equals(frame[right_name]):
                if left_name not in candidates:
                    candidates.append(left_name)
                if right_name not in candidates:
                    candidates.append(right_name)

    return candidates


def profile_dataset(
    csv_path: Annotated[
        str,
        Field(
            min_length=1,
            description="Absolute or working-directory-relative path to a CSV file.",
        ),
    ],
) -> ProfileResult:
    """Profile a CSV and return only schema, null, leakage, and hash metadata."""

    path = Path(csv_path).expanduser()
    if path.suffix.lower() != ".csv":
        raise ProfileDatasetError("csv_path must point to a .csv file")
    if not path.exists():
        raise ProfileDatasetError("CSV file does not exist")
    if not path.is_file():
        raise ProfileDatasetError("csv_path must point to a regular file")
    if path.stat().st_size >= MAX_CSV_BYTES:
        raise ProfileDatasetError(
            f"CSV file must be smaller than {MAX_CSV_BYTES // (1024 * 1024)} MiB"
        )

    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ProfileDatasetError("CSV file is empty or malformed") from exc
    except OSError as exc:
        raise ProfileDatasetError("CSV file could not be read") from exc

    if not len(frame.columns):
        raise ProfileDatasetError("CSV file must contain at least one column")
    if any(not str(column).strip() for column in frame.columns):
        raise ProfileDatasetError("CSV column names must not be blank")
    if len({str(column) for column in frame.columns}) != len(frame.columns):
        raise ProfileDatasetError("CSV column names must be unique")

    schema = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    null_counts = {
        str(column): int(count) for column, count in frame.isna().sum().items()
    }
    return ProfileResult(
        schema=schema,
        null_counts=null_counts,
        leakage_candidates=_find_leakage_candidates(frame),
        dataset_hash=_sha256(path),
    )


server = FastMCP("dataset-profile")
server.tool(
    name="profile_dataset",
    title="Profile CSV dataset",
    description=(
        "Inspect a local CSV and return schema, null counts, possible leakage "
        "columns, and a SHA-256 hash without returning raw rows."
    ),
)(profile_dataset)


def main() -> None:
    """Run the MCP server over stdio."""

    server.run()


if __name__ == "__main__":
    main()
