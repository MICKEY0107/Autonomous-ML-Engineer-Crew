"""Shared runtime helpers for the executable end-to-end demonstrations."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from contracts.models import PipelineResult, PipelineState, PipelineStatus, TaskType
from main import create_pipeline_state
from orchestrator.flow import FlowCallbackEvent, FlowCallbacks, build_flow
from orchestrator.llm_reasoning import generate_pipeline_summary
from orchestrator.telemetry import (
    flush_traces,
    get_tracer_provider,
    langfuse_trace_url,
)


DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
A2A_HOST = os.environ["A2A_BIND_HOST"]
A2A_PORT = int(os.environ["A2A_BIND_PORT"])
A2A_URL = os.environ["A2A_TASKS_URL"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    path: Path
    target_column: str
    task_type: TaskType
    prediction_goal: str
    source: str


@dataclass(slots=True)
class A2AServer:
    process: subprocess.Popen[str] | None
    log_file: Any | None
    log_path: Path

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log_file is not None:
            self.log_file.close()


@dataclass(frozen=True, slots=True)
class RunOutcome:
    state: PipelineState
    result: PipelineResult
    trace_exporter: InMemorySpanExporter
    elapsed_seconds: float


def _generate_house_prices(rows: int = 500) -> pd.DataFrame:
    randomizer = random.Random(20261022)
    neighborhood_effects = {
        "central": 135_000,
        "riverside": 90_000,
        "suburban": 45_000,
        "outskirts": 0,
    }
    property_effects = {
        "detached": 65_000,
        "townhouse": 30_000,
        "apartment": 0,
    }
    records: list[dict[str, object]] = []
    for _ in range(rows):
        neighborhood = randomizer.choices(
            list(neighborhood_effects),
            weights=[0.20, 0.22, 0.38, 0.20],
            k=1,
        )[0]
        property_type = randomizer.choices(
            list(property_effects),
            weights=[0.48, 0.24, 0.28],
            k=1,
        )[0]
        living_area = randomizer.randint(650, 3_800)
        lot_size = randomizer.randint(900, 12_000)
        bedrooms = max(1, min(6, round(living_area / 650 + randomizer.gauss(0, 0.7))))
        bathrooms = max(1.0, min(4.5, bedrooms * 0.55 + randomizer.choice([0, 0.5, 1])))
        year_built = randomizer.randint(1955, 2025)
        garage_spaces = randomizer.choices([0, 1, 2, 3], [0.12, 0.32, 0.46, 0.10], k=1)[0]
        has_garden = property_type != "apartment" and randomizer.random() < 0.66
        age = 2026 - year_built
        sale_price = (
            42_000
            + living_area * 185
            + lot_size * 5.5
            + bedrooms * 7_500
            + bathrooms * 15_000
            + garage_spaces * 18_000
            + (22_000 if has_garden else 0)
            + neighborhood_effects[neighborhood]
            + property_effects[property_type]
            - age * 1_050
            + randomizer.gauss(0, 18_000)
        )
        records.append(
            {
                "living_area_sqft": living_area,
                "lot_size_sqft": lot_size if randomizer.random() > 0.025 else None,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "year_built": year_built if randomizer.random() > 0.015 else None,
                "garage_spaces": garage_spaces,
                "has_garden": has_garden,
                "neighborhood": neighborhood,
                "property_type": property_type,
                "sale_price": round(max(sale_price, 85_000), 2),
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_demo_dataset() -> DatasetSpec:
    """Try Seaborn Titanic, then deterministically generate house-price data."""

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import seaborn as sns

        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(12)
        try:
            frame = sns.load_dataset("titanic")
        finally:
            socket.setdefaulttimeout(previous_timeout)
        columns = [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
            "alone",
            "survived",
        ]
        frame = frame[columns].copy()
        frame.to_csv(DATA_PATH, index=False)
        return DatasetSpec(
            path=DATA_PATH,
            target_column="survived",
            task_type=TaskType.CLASSIFICATION,
            prediction_goal="Predict whether a Titanic passenger survived.",
            source="seaborn:titanic",
        )
    except Exception as exc:
        print(
            f"Dataset download unavailable ({type(exc).__name__}); "
            "generated 500 realistic house-price rows instead.",
            flush=True,
        )

    frame = _generate_house_prices()
    frame.to_csv(DATA_PATH, index=False)
    return DatasetSpec(
        path=DATA_PATH,
        target_column="sale_price",
        task_type=TaskType.REGRESSION,
        prediction_goal="Predict residential sale price from property attributes.",
        source="generated:house_prices",
    )


def _port_is_open() -> bool:
    try:
        with socket.create_connection((A2A_HOST, A2A_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def start_a2a_server(run_name: str) -> A2AServer:
    """Start the ADK service unless an A2A listener is already available."""

    log_path = PROJECT_ROOT / "artifacts" / run_name / "a2a_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _port_is_open():
        print(f"Using existing A2A service at {A2A_URL}", flush=True)
        return A2AServer(process=None, log_file=None, log_path=log_path)

    log_file = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [sys.executable, "-m", "agents.model_selection_adk"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creation_flags,
    )
    server = A2AServer(process=process, log_file=log_file, log_path=log_path)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.flush()
            details = log_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
            server.stop()
            raise RuntimeError(f"A2A service exited during startup:\n{details}")
        if _port_is_open():
            print(f"A2A service ready at {A2A_URL}", flush=True)
            return server
        time.sleep(0.25)
    server.stop()
    raise TimeoutError(f"A2A service did not start; inspect {log_path}")


def _callback(event: FlowCallbackEvent) -> None:
    print(
        "FLOW_EVENT "
        + json.dumps(
            {
                "name": event.name,
                "stage": event.stage,
                "retry_count": event.retry_count,
                "status": event.status,
                "message": event.message,
            },
            default=str,
        ),
        flush=True,
    )


def _callbacks() -> FlowCallbacks:
    return FlowCallbacks(
        on_stage_start=_callback,
        on_stage_complete=_callback,
        on_stage_error=_callback,
        on_retry=_callback,
        on_validation_failure=_callback,
    )


def _trace_url(run_id: UUID) -> tuple[str, bool]:
    configured = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    if configured:
        return langfuse_trace_url(run_id), True
    return f"http://localhost/traces/{run_id.hex}", False


def _pipeline_result(
    state: PipelineState,
    elapsed_seconds: float,
    *,
    llm_summary: str | None = None,
) -> PipelineResult:
    trace_url, _ = _trace_url(state.run_id)
    lineage: dict[str, str] = {"pipeline_run_id": str(state.run_id)}
    if state.data_prep_result is not None:
        lineage["dataset_sha256"] = state.data_prep_result.dataset_hash
        lineage["train_artifact"] = state.data_prep_result.train_artifact_uri
        lineage["test_artifact"] = state.data_prep_result.test_artifact_uri
    if state.model_selection_result is not None:
        lineage["model_run_id"] = state.model_selection_result.best_run_id
        lineage["model_uri"] = state.model_selection_result.best_model_uri

    succeeded = state.status == PipelineStatus.SUCCESS
    error = None if succeeded else "; ".join(state.errors) or str(state.status)
    evaluation = state.evaluation_result
    metrics = evaluation.held_out_metrics if evaluation is not None else {}
    model_card = (
        "# Autonomous ML pipeline result\n\n"
        f"- Terminal state: `{state.status}`\n"
        f"- Task type: `{state.task_type}`\n"
        f"- Target: `{state.target_column}`\n"
        f"- Retry count: `{state.retry_count}`\n"
        f"- Held-out metrics: `{json.dumps(metrics, sort_keys=True)}`\n"
    )
    selection = state.model_selection_result
    metric_name = evaluation.primary_metric if evaluation is not None else "metric"
    metric_value = (
        float(evaluation.held_out_metrics[metric_name])
        if evaluation is not None
        else 0.0
    )
    summary = llm_summary or generate_pipeline_summary(
        str(state.status),
        selection.best_model_name if selection is not None else "unknown",
        metric_name,
        metric_value,
        state.retry_count,
        elapsed_seconds,
    )
    return PipelineResult(
        run_id=state.run_id,
        status="evaluation_completed" if succeeded else "failed",
        model_card_markdown=model_card,
        lineage=lineage,
        trace_url=trace_url,
        evaluation=evaluation,
        runtime_seconds=elapsed_seconds,
        error=error,
        llm_summary=summary,
    )


def run_pipeline(
    dataset: DatasetSpec,
    *,
    run_name: str,
    evaluation_threshold: float | None = None,
) -> RunOutcome:
    exporter = InMemorySpanExporter()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    state = create_pipeline_state(
        csv_path=dataset.path,
        target_column=dataset.target_column,
        task_type=dataset.task_type,
        prediction_goal=dataset.prediction_goal,
    )
    artifact_dir = PROJECT_ROOT / "artifacts" / run_name
    tracking_database = (artifact_dir / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{tracking_database.as_posix()}"
    flow = build_flow(
        state,
        callbacks=_callbacks(),
        a2a_url=A2A_URL,
        tracking_uri=tracking_uri,
        artifact_dir=artifact_dir,
        evaluation_threshold=evaluation_threshold,
        suppress_flow_events=True,
    )
    started = time.perf_counter()
    flow.kickoff()
    elapsed = time.perf_counter() - started
    result = _pipeline_result(
        flow.state,
        elapsed,
        llm_summary=flow.llm_summary,
    )
    flush_traces()
    return RunOutcome(
        state=flow.state,
        result=result,
        trace_exporter=exporter,
        elapsed_seconds=elapsed,
    )


def print_stage_results(outcome: RunOutcome) -> None:
    for name, value in (
        ("DataPrepResult", outcome.state.data_prep_result),
        ("ModelSelectionResult", outcome.state.model_selection_result),
        ("EvaluationResult", outcome.state.evaluation_result),
    ):
        print(f"\n=== {name} ===")
        print(value.model_dump_json(indent=2) if value is not None else "null")
    print("\n=== PipelineResult ===")
    print(outcome.result.model_dump_json(indent=2))


def print_trace_location(outcome: RunOutcome) -> None:
    trace_url, configured = _trace_url(outcome.state.run_id)
    if configured:
        print(f"\nLangfuse trace URL: {trace_url}")
    else:
        print("\ntraces stored locally")


def print_terminal_failure(outcome: RunOutcome) -> None:
    if outcome.state.status != PipelineStatus.SUCCESS:
        reason = "; ".join(outcome.state.errors) or "No reason was recorded."
        print(f"\nTerminal state: {outcome.state.status}")
        print(f"Reason: {reason}")


def print_retry_spans(outcome: RunOutcome) -> int:
    attempts = sorted(
        (
            span
            for span in outcome.trace_exporter.get_finished_spans()
            if span.name == "agent.model_selection"
        ),
        key=lambda span: span.start_time,
    )
    print(f"\n=== Model-selection attempt spans ({len(attempts)}) ===")
    for span in attempts:
        print(
            json.dumps(
                {
                    "name": span.name,
                    "trace_id": f"{span.context.trace_id:032x}",
                    "span_id": f"{span.context.span_id:016x}",
                    "parent_span_id": (
                        f"{span.parent.span_id:016x}" if span.parent else None
                    ),
                    "retry_count": span.attributes.get("ml.retry_count"),
                    "status": span.status.status_code.name,
                },
                indent=2,
            )
        )
    return len(attempts)
