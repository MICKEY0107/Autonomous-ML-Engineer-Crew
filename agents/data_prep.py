"""CrewAI data-preparation agent and its deterministic execution tool."""

from __future__ import annotations

import asyncio
import contextvars
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pandas as pd
from crewai import Agent
from crewai.tools import BaseTool
from mcp.shared.memory import create_connected_server_and_client_session
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.model_selection import train_test_split

from contracts.models import DataPrepResult, TaskType
from tools.profile_dataset import (
    MAX_CSV_BYTES,
    ProfileResult,
    server as profile_mcp_server,
)
from orchestrator.agentic_policy import plan_data_prep
from orchestrator.telemetry import (
    set_current_span_attributes,
    traced_span,
    update_span_io,
)


DEFAULT_ARTIFACT_DIR = Path("/tmp/artifacts")
MIN_CSV_ROWS = 11
_FORBIDDEN_ROW_KEYS = frozenset(
    {
        "dataframe",
        "dataset_rows",
        "raw_data",
        "raw_rows",
        "records",
        "rows",
        "samples",
        "test_data",
        "training_data",
    }
)
_LLM_SAFE_DATA_PREP_FIELDS = frozenset(DataPrepResult.model_fields)


class DataPrepError(ValueError):
    """Safe error raised when a dataset cannot be prepared."""


def validate_csv_input(csv_path: str | Path) -> Path:
    """Validate an input path before resolving or reading any CSV bytes."""

    raw_path = str(csv_path).strip()
    if not raw_path:
        raise DataPrepError("csv_path must not be blank")
    if "\x00" in raw_path:
        raise DataPrepError("csv_path contains an invalid null byte")
    if ".." in re.split(r"[\\\\/]+", raw_path):
        raise DataPrepError("csv_path must not contain path traversal segments")

    path = Path(raw_path).expanduser().resolve()
    if path.suffix.lower() != ".csv":
        raise DataPrepError("csv_path must point to a .csv file")
    if not path.exists():
        raise DataPrepError("CSV file does not exist")
    if not path.is_file():
        raise DataPrepError("csv_path must point to a regular file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DataPrepError("CSV file metadata could not be read") from exc
    if size >= MAX_CSV_BYTES:
        raise DataPrepError("CSV file must be smaller than 100 MiB")
    try:
        sample = pd.read_csv(path, nrows=MIN_CSV_ROWS)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataPrepError("CSV file is empty or malformed") from exc
    except OSError as exc:
        raise DataPrepError("CSV file could not be read") from exc
    if len(sample) < MIN_CSV_ROWS:
        raise DataPrepError("CSV file must contain more than 10 data rows")
    return path


def validate_profile_mcp_result(payload: Any) -> ProfileResult:
    """Fail closed unless an MCP payload exactly matches ``ProfileResult``."""

    if not isinstance(payload, Mapping):
        raise DataPrepError("profile_dataset MCP result must be a JSON object")
    try:
        return ProfileResult.model_validate(dict(payload), strict=True)
    except ValidationError as exc:
        raise DataPrepError(
            "profile_dataset MCP result failed contract validation"
        ) from exc


def assert_no_raw_rows(payload: BaseModel | Mapping[str, Any]) -> None:
    """Assert that an LLM-facing payload contains metadata, never row values."""

    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    if not isinstance(data, Mapping):
        raise DataPrepError("LLM payload must be a metadata object")
    unexpected = set(data) - _LLM_SAFE_DATA_PREP_FIELDS
    if unexpected:
        raise DataPrepError(
            "LLM payload contains non-metadata fields: " + ", ".join(sorted(unexpected))
        )

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
                if normalized_key in _FORBIDDEN_ROW_KEYS:
                    raise DataPrepError(
                        f"raw dataset rows are forbidden in LLM payload field '{key}'"
                    )
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif isinstance(value, pd.DataFrame):
            raise DataPrepError("pandas DataFrames are forbidden in LLM payloads")

    walk(data)


class DataPrepToolInput(BaseModel):
    """Validated input exposed to the CrewAI tool."""

    model_config = ConfigDict(extra="forbid")

    csv_path: str = Field(min_length=1, description="Path to the source CSV file.")
    target_column: str = Field(min_length=1, description="Column to predict.")
    run_id: UUID = Field(description="Pipeline run identifier.")
    task_type: TaskType = Field(description="classification or regression")
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_seed: int = Field(default=42, ge=0)


async def _profile_via_mcp(csv_path: Path) -> ProfileResult:
    """Call the profile tool through an MCP-validated in-memory session."""

    with traced_span(
        "mcp.profile_dataset",
        kind=SpanKind.CLIENT,
        attributes={
            "rpc.system": "mcp",
            "rpc.method": "profile_dataset",
            "mcp.transport": "memory",
            "ml.dataset_uri": csv_path.as_uri(),
        },
    ) as span:
        update_span_io(span, input={"csv_path": str(csv_path)})
        async with create_connected_server_and_client_session(
            profile_mcp_server,
            raise_exceptions=True,
        ) as session:
            response = await session.call_tool(
                "profile_dataset",
                {"csv_path": str(csv_path)},
            )
        structured_content = getattr(
            response,
            "structuredContent",
            getattr(response, "structured_content", None),
        )
        is_error = getattr(response, "isError", getattr(response, "is_error", False))
        if is_error or structured_content is None:
            raise DataPrepError("profile_dataset MCP tool did not return a valid result")
        result = validate_profile_mcp_result(structured_content)
        set_current_span_attributes(
            {
                "mcp.result.validated": True,
                "ml.schema": result.schema,
                "ml.null_counts": result.null_counts,
                "ml.leakage_candidate_count": len(result.leakage_candidates),
            }
        )
        update_span_io(
            span,
            output={
                "sha256": result.sha256,
                "null_count": sum(result.null_counts.values()),
                "feature_count": len(result.schema),
            },
        )
        return result


def _run_async_profile(csv_path: Path) -> ProfileResult:
    """Run the MCP call from synchronous CrewAI and Flow entry points."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_profile_via_mcp(csv_path))

    # A synchronous tool can be invoked while an async orchestration loop is
    # active. A short-lived worker avoids nesting asyncio.run in that loop.
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(
            context.run,
            asyncio.run,
            _profile_via_mcp(csv_path),
        ).result()


def _usable_stratification_target(
    target: pd.Series,
    task_type: TaskType,
    test_size: float,
) -> pd.Series | None:
    if task_type != TaskType.CLASSIFICATION:
        return None
    class_counts = target.value_counts(dropna=False)
    class_count = len(class_counts)
    test_rows = math.ceil(len(target) * test_size)
    train_rows = len(target) - test_rows
    if (
        class_count < 2
        or class_counts.min() < 2
        or test_rows < class_count
        or train_rows < class_count
    ):
        return None
    return target


def _clean_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    quality_findings: list[str],
    *,
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute using training-only statistics to prevent split leakage."""

    for column in feature_columns:
        train_series = train[column]
        if pd.api.types.is_numeric_dtype(train_series):
            fill_value = (
                train_series.mean()
                if numeric_imputation == "mean"
                else train_series.median()
            )
            if pd.isna(fill_value):
                fill_value = 0
            missing_count = int(train[column].isna().sum() + test[column].isna().sum())
            train[column] = train[column].fillna(fill_value)
            test[column] = test[column].fillna(fill_value)
        else:
            modes = train_series.mode(dropna=True)
            fill_value = (
                "__missing__"
                if categorical_imputation == "constant"
                else (str(modes.iloc[0]) if not modes.empty else "__missing__")
            )
            missing_count = int(train[column].isna().sum() + test[column].isna().sum())
            train[column] = train[column].fillna(fill_value)
            test[column] = test[column].fillna(fill_value)
        if missing_count:
            quality_findings.append(
                f"Imputed {missing_count} missing value(s) in '{column}' "
                "using training-only statistics."
            )
    return train, test


def _write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def prepare_dataset(
    csv_path: str | Path,
    target_column: str,
    run_id: UUID | str,
    task_type: TaskType | str,
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    prediction_goal: str = "",
) -> DataPrepResult:
    """Profile, clean, split, and persist a tabular supervised-ML dataset."""

    try:
        validated_run_id = UUID(str(run_id))
    except ValueError as exc:
        raise DataPrepError("run_id must be a valid UUID") from exc
    try:
        validated_task_type = TaskType(task_type)
    except ValueError as exc:
        raise DataPrepError("task_type must be classification or regression") from exc
    if not 0 < test_size < 1:
        raise DataPrepError("test_size must be between 0 and 1")
    if random_seed < 0:
        raise DataPrepError("random_seed cannot be negative")

    source_path = validate_csv_input(csv_path)
    profile = _run_async_profile(source_path)
    if target_column not in profile.schema:
        raise DataPrepError("target_column is not present in the CSV schema")

    frame = pd.read_csv(source_path)
    source_row_count = len(frame)
    if source_row_count < MIN_CSV_ROWS:
        raise DataPrepError("CSV file must contain more than 10 data rows")
    prep_plan = plan_data_prep(
        schema=profile.schema,
        null_counts=profile.null_counts,
        leakage_candidates=profile.leakage_candidates,
        row_count=source_row_count,
        target_column=target_column,
        prediction_goal=(
            prediction_goal or f"prepare {target_column} for supervised prediction"
        ),
        requested_test_size=test_size,
    )
    effective_test_size = prep_plan.test_size
    quality_findings = list(profile.leakage_candidates)
    quality_findings = [
        f"Potential leakage column detected: '{column}'."
        for column in quality_findings
    ]

    missing_target_rows = int(frame[target_column].isna().sum())
    if missing_target_rows:
        frame = frame.dropna(subset=[target_column]).copy()
        quality_findings.append(
            f"Dropped {missing_target_rows} row(s) with a missing target."
        )

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        frame = frame.drop_duplicates().copy()
        quality_findings.append(f"Dropped {duplicate_rows} duplicate row(s).")

    leakage_features = {
        column for column in profile.leakage_candidates if column != target_column
    }
    llm_exclusions = {
        column
        for column in prep_plan.additional_exclusions
        if column != target_column
    }
    leakage_features.update(llm_exclusions)
    candidate_features = [
        str(column)
        for column in frame.columns
        if column != target_column and str(column) not in leakage_features
    ]
    if leakage_features:
        quality_findings.append(
            "Excluded leakage candidate feature(s): "
            + ", ".join(sorted(leakage_features))
            + "."
        )
    quality_findings.append(
        "Agentic preparation plan applied: "
        f"test_size={effective_test_size:.2f}, "
        f"numeric_imputation={prep_plan.numeric_imputation}, "
        f"categorical_imputation={prep_plan.categorical_imputation}."
    )

    for column in candidate_features:
        if pd.api.types.is_numeric_dtype(frame[column]):
            infinite_count = int(frame[column].isin([math.inf, -math.inf]).sum())
            if infinite_count:
                frame[column] = frame[column].replace([math.inf, -math.inf], float("nan"))
                quality_findings.append(
                    f"Converted {infinite_count} infinite value(s) in '{column}' to missing."
                )

    unusable_features = [
        column
        for column in candidate_features
        if frame[column].isna().all() or frame[column].nunique(dropna=False) <= 1
    ]
    feature_columns = [
        column for column in candidate_features if column not in unusable_features
    ]
    if unusable_features:
        quality_findings.append(
            "Dropped empty or constant feature(s): "
            + ", ".join(sorted(unusable_features))
            + "."
        )
    if not feature_columns:
        raise DataPrepError("no usable feature columns remain after leakage checks")
    if len(frame) < 2:
        raise DataPrepError("at least two usable rows are required for splitting")

    prepared = frame[[*feature_columns, target_column]].copy()
    stratify = _usable_stratification_target(
        prepared[target_column],
        validated_task_type,
        effective_test_size,
    )
    try:
        train, test = train_test_split(
            prepared,
            test_size=effective_test_size,
            random_state=random_seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        raise DataPrepError("dataset cannot be split with the requested settings") from exc
    if validated_task_type == TaskType.CLASSIFICATION and stratify is None:
        quality_findings.append(
            "Classification split was not stratified because class counts were too small."
        )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train, test = _clean_features(
        train,
        test,
        feature_columns,
        quality_findings,
        numeric_imputation=prep_plan.numeric_imputation,
        categorical_imputation=prep_plan.categorical_imputation,
    )

    run_artifact_dir = Path(artifact_dir).expanduser().resolve() / str(validated_run_id)
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    train_path = run_artifact_dir / "train.csv"
    test_path = run_artifact_dir / "test.csv"
    _write_csv_atomically(train, train_path)
    _write_csv_atomically(test, test_path)

    schema_summary = {
        str(column): str(dtype)
        for column, dtype in pd.concat([train, test], ignore_index=True).dtypes.items()
    }
    if not quality_findings:
        quality_findings.append("No data-quality issues were detected.")

    result = DataPrepResult(
        run_id=validated_run_id,
        dataset_hash=profile.dataset_hash,
        schema_summary=schema_summary,
        quality_findings=quality_findings,
        target_column=target_column,
        feature_columns=feature_columns,
        row_count=source_row_count,
        train_row_count=len(train),
        test_row_count=len(test),
        train_artifact_uri=train_path.as_uri(),
        test_artifact_uri=test_path.as_uri(),
        llm_reasoning=prep_plan.reasoning,
    )
    assert_no_raw_rows(result)
    return result


def run_data_prep(
    csv_path: str | Path,
    target_column: str,
    run_id: UUID | str,
    *,
    task_type: TaskType | str | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    prediction_goal: str = "",
) -> DataPrepResult:
    """Prepare data using either a UUID or a human-readable run label."""

    try:
        normalized_run_id = UUID(str(run_id))
    except ValueError:
        if not str(run_id).strip():
            raise DataPrepError("run_id must not be blank") from None
        normalized_run_id = uuid5(NAMESPACE_URL, f"autonomous-ml:{run_id}")

    resolved_task_type = task_type
    if resolved_task_type is None:
        source_path = validate_csv_input(csv_path)
        source = pd.read_csv(
            source_path,
            usecols=[target_column],
        )
        target = source[target_column].dropna()
        unique_count = target.nunique()
        resolved_task_type = (
            TaskType.CLASSIFICATION
            if unique_count <= max(20, int(len(target) * 0.1))
            else TaskType.REGRESSION
        )
    return prepare_dataset(
        csv_path=csv_path,
        target_column=target_column,
        run_id=normalized_run_id,
        task_type=resolved_task_type,
        test_size=test_size,
        random_seed=random_seed,
        artifact_dir=artifact_dir,
        prediction_goal=prediction_goal,
    )


class PrepareDatasetTool(BaseTool):
    """CrewAI tool that executes deterministic preparation after MCP profiling."""

    name: str = "prepare_dataset"
    description: str = (
        "Call the profile_dataset MCP tool, clean a supervised-learning CSV, "
        "split it into train and test artifacts, and return DataPrepResult JSON."
    )
    args_schema: type[BaseModel] = DataPrepToolInput
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR)

    def _run(
        self,
        csv_path: str,
        target_column: str,
        run_id: UUID,
        task_type: TaskType,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> str:
        result = prepare_dataset(
            csv_path=csv_path,
            target_column=target_column,
            run_id=run_id,
            task_type=task_type,
            test_size=test_size,
            random_seed=random_seed,
            artifact_dir=self.artifact_dir,
        )
        assert_no_raw_rows(result)
        return result.model_dump_json()


def build_data_prep_agent(
    *,
    llm: Any | None = None,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> Agent:
    """Construct the CrewAI agent without making an LLM call at import time."""

    kwargs: dict[str, Any] = {
        "role": "Data Preparation Engineer",
        "goal": (
            "Profile CSV data through MCP, prevent target leakage, and produce "
            "reproducible clean train/test artifacts with a typed hand-off."
        ),
        "backstory": (
            "You are a cautious tabular-ML data engineer. You never expose raw "
            "rows to the language model; you use the prepare_dataset tool and "
            "return its validated DataPrepResult without inventing fields."
        ),
        "tools": [PrepareDatasetTool(artifact_dir=str(artifact_dir))],
        "allow_delegation": False,
        "verbose": False,
        "max_iter": 3,
    }
    if llm is not None:
        kwargs["llm"] = llm
    return Agent(**kwargs)
