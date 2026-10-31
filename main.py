"""Guarded command-line entry point for the autonomous ML pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
warnings.filterwarnings(
    "ignore",
    message=r"Field 'lifespan' has an incomplete definition.*",
)
warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\].*")

import pandas as pd
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)
from pydantic import BaseModel

from agents.data_prep import DataPrepError, validate_csv_input
from contracts.models import PipelineResult, PipelineState, PipelineStatus, TaskType
from orchestrator.flow import (
    DEFAULT_A2A_URL,
    MAX_EVALUATION_RETRIES,
    FlowCallbacks,
    build_flow,
)
from orchestrator.telemetry import (
    flush_traces,
    get_tracer_provider,
    langfuse_trace_url,
)
from orchestrator import usage
from orchestrator.llm_reasoning import generate_pipeline_summary


MAX_PREDICTION_GOAL_CHARS = 2_000
PROMPT_INJECTION_KEYWORDS = frozenset(
    {
        "bypass safety",
        "developer message",
        "disregard previous",
        "execute shell",
        "forget previous",
        "ignore all instructions",
        "ignore previous instructions",
        "jailbreak",
        "override system",
        "prompt injection",
        "reveal secrets",
        "run command",
        "system prompt",
    }
)
PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override)\b.{0,80}"
        r"\b(?:instruction|message|policy|prompt|rule)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(?m)^\s*(?:assistant|developer|system)\s*:", re.IGNORECASE),
    re.compile(r"<\|\s*(?:assistant|developer|system|user)\s*\|>", re.IGNORECASE),
    re.compile(r"\b(?:show|reveal|print|leak)\b.{0,50}\bsystem prompt\b", re.IGNORECASE),
)
# Toxic / abusive-language guardrail (requirement 8). Word-boundary matching on
# the normalised goal, so ML phrases such as "kill rate" or "attack surface"
# are not blocked; only profanity, abuse and threats aimed at people.
TOXIC_TERMS = frozenset(
    {
        "asshole",
        "bastard",
        "bitch",
        "bullshit",
        "dickhead",
        "fuck",
        "fucking",
        "motherfucker",
        "retard",
        "shit",
        "slut",
        "whore",
    }
)
TOXIC_PATTERNS = (
    re.compile(
        r"\b(?:kill|hurt|murder|rape|shoot|stab|attack)\s+"
        r"(?:you|him|her|them|people|everyone|myself|yourself)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i|we)\s+(?:hate|despise)\s+(?:you|all|them|those)\b", re.IGNORECASE),
    re.compile(r"\b(?:go\s+die|kys)\b", re.IGNORECASE),
    re.compile(r"\b(?:stupid|dumb|useless)\s+(?:bot|ai|idiot|moron)\b", re.IGNORECASE),
)
_TOXIC_WORD = re.compile(r"[a-z]+")


def find_toxic_language(text: str) -> list[str]:
    """Return the toxicity categories found in ``text`` (empty when clean)."""

    normalized = " ".join(str(text).casefold().split())
    found: list[str] = []
    if any(word in TOXIC_TERMS for word in _TOXIC_WORD.findall(normalized)):
        found.append("profanity_or_abuse")
    if any(pattern.search(normalized) for pattern in TOXIC_PATTERNS):
        found.append("threat_or_harassment")
    return found


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
_SAFE_PROMPT_FIELDS = frozenset(
    {
        "artifact_uris",
        "dataset_uri",
        "evaluation_metrics",
        "prediction_goal",
        "retry_feedback",
        "run_id",
        "schema",
        "target_column",
        "task_type",
        "training_metrics",
    }
)


class GuardrailViolation(ValueError):
    """Raised when untrusted input fails a pipeline boundary guardrail."""


def validate_prediction_goal(prediction_goal: str) -> str:
    """Reject empty, oversized, or prompt-injection-like prediction goals."""

    goal = str(prediction_goal).strip()
    if not goal:
        raise GuardrailViolation("prediction_goal must not be blank")
    if len(goal) > MAX_PREDICTION_GOAL_CHARS:
        raise GuardrailViolation(
            f"prediction_goal must not exceed {MAX_PREDICTION_GOAL_CHARS} characters"
        )

    normalized = " ".join(goal.casefold().split())
    matched_keyword = next(
        (keyword for keyword in PROMPT_INJECTION_KEYWORDS if keyword in normalized),
        None,
    )
    if matched_keyword or any(
        pattern.search(goal) for pattern in PROMPT_INJECTION_PATTERNS
    ):
        raise GuardrailViolation(
            "prediction_goal contains prompt-injection-like instructions"
        )
    toxic = find_toxic_language(goal)
    if toxic:
        raise GuardrailViolation(
            "prediction_goal contains toxic or abusive language "
            f"({', '.join(toxic)})"
        )
    return goal


def assert_llm_prompt_safe(payload: BaseModel | Mapping[str, Any]) -> None:
    """Fail if a prompt context contains raw rows or non-metadata fields."""

    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    if not isinstance(data, Mapping):
        raise GuardrailViolation("LLM prompt context must be a metadata object")
    unexpected = set(data) - _SAFE_PROMPT_FIELDS
    if unexpected:
        raise GuardrailViolation(
            "LLM prompt context contains non-metadata fields: "
            + ", ".join(sorted(str(key) for key in unexpected))
        )

    def walk(value: Any) -> None:
        if isinstance(value, pd.DataFrame):
            raise GuardrailViolation("pandas DataFrames cannot be passed to an LLM")
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = re.sub(
                    r"[^a-z0-9]+", "_", str(key).casefold()
                ).strip("_")
                if normalized_key in _FORBIDDEN_ROW_KEYS:
                    raise GuardrailViolation(
                        f"raw dataset rows are forbidden in LLM prompt field '{key}'"
                    )
                walk(child)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if any(isinstance(item, Mapping) for item in value):
                raise GuardrailViolation(
                    "lists of records cannot be passed to an LLM prompt"
                )
            for item in value:
                walk(item)

    walk(data)


def build_llm_safe_context(state: PipelineState) -> dict[str, Any]:
    """Build the only context shape allowed to cross an LLM boundary."""

    context: dict[str, Any] = {
        "run_id": str(state.run_id),
        "dataset_uri": state.dataset_uri,
        "task_type": str(state.task_type),
        "target_column": state.target_column,
        "prediction_goal": validate_prediction_goal(state.prediction_goal),
        "artifact_uris": dict(state.artifact_references),
    }
    if state.data_prep_result is not None:
        context["schema"] = dict(state.data_prep_result.schema_summary)
        context["artifact_uris"] = {
            **context["artifact_uris"],
            "train": state.data_prep_result.train_artifact_uri,
            "test": state.data_prep_result.test_artifact_uri,
        }
    if state.model_selection_result is not None:
        context["training_metrics"] = dict(
            state.model_selection_result.training_metrics
        )
    if state.evaluation_result is not None:
        context["evaluation_metrics"] = dict(
            state.evaluation_result.held_out_metrics
        )
        if state.evaluation_result.retry_feedback:
            context["retry_feedback"] = state.evaluation_result.retry_feedback
    assert_llm_prompt_safe(context)
    return context


def create_pipeline_state(
    *,
    csv_path: str | Path,
    target_column: str,
    task_type: TaskType | str,
    prediction_goal: str,
) -> PipelineState:
    """Validate external inputs before creating mutable orchestration state."""

    source_path = validate_csv_input(csv_path)
    safe_goal = validate_prediction_goal(prediction_goal)
    run_id = uuid4()
    state = PipelineState(
        run_id=run_id,
        dataset_uri=source_path.as_uri(),
        task_type=TaskType(task_type),
        target_column=target_column,
        prediction_goal=safe_goal,
        trace_id=run_id.hex,
    )
    build_llm_safe_context(state)
    return state


def infer_task_type(csv_path: str | Path, target_column: str) -> TaskType:
    """Infer classification versus regression from a validated target column."""

    source_path = validate_csv_input(csv_path)
    target = str(target_column).strip()
    if not target:
        raise DataPrepError("target column must not be blank")
    try:
        series = pd.read_csv(source_path, usecols=[target])[target].dropna()
    except ValueError as exc:
        raise DataPrepError(f"target column '{target}' does not exist") from exc
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataPrepError("CSV target column could not be read") from exc
    if series.empty:
        raise DataPrepError("target column must contain at least one non-null value")
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return TaskType.CLASSIFICATION

    unique_count = int(series.nunique())
    classification_limit = min(50, max(20, int(len(series) ** 0.5)))
    numeric = pd.to_numeric(series, errors="raise")
    integer_like = bool(((numeric % 1) == 0).all())
    if integer_like and unique_count <= classification_limit:
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION


def _format_metric(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000:
        return f"{value:,.2f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_model_name(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^a-zA-Z0-9]+", value))


def _span_payload(span: ReadableSpan, field: str) -> dict[str, Any]:
    encoded = (span.attributes or {}).get(f"langfuse.observation.{field}")
    if not isinstance(encoded, str):
        return {}
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _leakage_risk_count(output: Mapping[str, Any]) -> int:
    target = str(output.get("target_column", ""))
    candidates: set[str] = set()
    for finding in output.get("quality_findings", []):
        match = re.search(
            r"Potential leakage column detected: ['\"]([^'\"]+)['\"]",
            str(finding),
        )
        if match and match.group(1) != target:
            candidates.add(match.group(1))
    return len(candidates)


class _ProgressExporter(SpanExporter):
    """Render completed Flow spans as concise, truthful CLI progress."""

    def __init__(self, *, force_fail_eval: bool) -> None:
        self.force_fail_eval = force_fail_eval
        self.failed_evaluations = 0
        self.leakage_risk_count = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            span_input = _span_payload(span, "input")
            span_output = _span_payload(span, "output")
            attributes = span.attributes or {}
            if span.name == "mcp.profile_dataset":
                self.leakage_risk_count = int(
                    attributes.get("ml.leakage_candidate_count", 0)
                )
            elif span.name == "agent.data_prep" and span_output:
                print(
                    "[DATA-PREP]   ✓ Profiled "
                    f"{span_output['row_count']} rows, "
                    f"{int(attributes.get('ml.feature_count', 0))} features, "
                    f"{self.leakage_risk_count} leakage risks",
                    flush=True,
                )
            elif span.name == "agent.model_selection" and span_output:
                attempt = int(span_input.get("retry_count", 0)) + 1
                suffix = "attempt" if attempt == 1 else "attempts"
                metric_name = str(attributes.get("ml.metric_name", "metric"))
                print(
                    "[MODEL-SEL]   ✓ Best model: "
                    f"{_format_model_name(str(span_output['best_model_name']))} "
                    f"({metric_name}="
                    f"{_format_metric(float(span_output['metric_value']))}) "
                    f"after {attempt} {suffix}",
                    flush=True,
                )
            elif span.name == "agent.evaluation" and span_output:
                self._print_evaluation(
                    span_input,
                    span_output,
                    str(attributes.get("ml.metric_name", "metric")),
                )
            elif span.name.startswith(("llm.reasoning.", "llm.decision.")) and span_output.get(
                "reasoning"
            ):
                label = (
                    span.name.removeprefix("llm.reasoning.")
                    .removeprefix("llm.decision.")
                    .replace("_", " ")
                    .title()
                )
                # Show where the text really came from, not a fixed provider name.
                source = span_output.get("source")
                if span_output.get("fallback_used"):
                    tag = "[FALLBACK]"
                elif source == "cache":
                    tag = "[CACHE]"
                elif source == "partial_fallback":
                    tag = "[PARTIAL]"
                elif source == "narrative_llm":
                    tag = "[LOCAL-LLM]"
                elif source == "secondary_llm":
                    tag = "[LLM-2ND]"
                else:
                    tag = f"[{os.getenv('LLM_PROVIDER', 'llm').upper()}]"
                print(
                    f"{tag:<12} {label}\n{span_output['reasoning']}",
                    flush=True,
                )
        return SpanExportResult.SUCCESS

    def _print_evaluation(
        self,
        span_input: Mapping[str, Any],
        span_output: Mapping[str, Any],
        metric_name: str,
    ) -> None:
        metric_value = float(span_input["metric_value"])
        threshold = float(span_input["threshold"])
        passed = bool(span_output["passed"])
        minimize = metric_name.casefold() in {"mae", "mse", "rmse", "log_loss"}
        comparison = (
            "is below"
            if (passed and minimize) or (not passed and not minimize)
            else "exceeds"
        )
        symbol = "✓" if passed else "✗"
        status = "PASSED" if passed else "FAILED"
        retry_suffix = ""
        if not passed:
            self.failed_evaluations += 1
            if (
                self.force_fail_eval
                and self.failed_evaluations > MAX_EVALUATION_RETRIES
            ):
                return
            retry_suffix = (
                f" | retry {self.failed_evaluations}/{MAX_EVALUATION_RETRIES}"
            )
        print(
            f"[EVAL]        {symbol} {status} — "
            f"{metric_name}={_format_metric(metric_value)} {comparison} "
            f"threshold {_format_metric(threshold)}{retry_suffix}",
            flush=True,
        )


@dataclass(slots=True)
class _A2AServer:
    process: subprocess.Popen[str] | None
    log_file: Any | None

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


def _a2a_port_is_open() -> bool:
    try:
        with socket.create_connection(
            ("127.0.0.1", int(os.environ["A2A_BIND_PORT"])), timeout=0.5
        ):
            return True
    except OSError:
        return False


def _start_a2a_server(artifact_dir: Path) -> _A2AServer:
    if _a2a_port_is_open():
        return _A2AServer(process=None, log_file=None)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "a2a_server.log"
    log_file = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    project_root = Path(__file__).resolve().parent
    environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "agents.model_selection_adk"],
        cwd=project_root,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    server = _A2AServer(process=process, log_file=log_file)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.flush()
            details = log_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
            server.stop()
            raise RuntimeError(f"A2A service failed to start: {details}")
        if _a2a_port_is_open():
            return server
        time.sleep(0.25)
    server.stop()
    raise TimeoutError(f"A2A service did not start; inspect {log_path}")


def _pipeline_result(
    state: PipelineState,
    runtime_seconds: float,
    *,
    llm_summary: str | None = None,
) -> PipelineResult:
    succeeded = state.status == PipelineStatus.SUCCESS
    error = None if succeeded else "; ".join(state.errors) or str(state.status)
    lineage: dict[str, str] = {"pipeline_run_id": str(state.run_id)}
    if state.data_prep_result is not None:
        lineage.update(
            {
                "dataset_sha256": state.data_prep_result.dataset_hash,
                "train_artifact": state.data_prep_result.train_artifact_uri,
                "test_artifact": state.data_prep_result.test_artifact_uri,
            }
        )
    if state.model_selection_result is not None:
        lineage.update(
            {
                "model_run_id": state.model_selection_result.best_run_id,
                "model_uri": state.model_selection_result.best_model_uri,
            }
        )
    selection = state.model_selection_result
    evaluation = state.evaluation_result
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
        runtime_seconds,
    )
    usage_report = usage.summary()
    deployment = state.deployment_result
    deployed = (
        succeeded
        and deployment is not None
        and str(deployment.health_status) == "healthy"
    )
    if deployed:
        lineage.update(
            {
                "deployment_revision": deployment.deployment_revision,
                "deployment_digest": deployment.image_digest,
            }
        )
    return PipelineResult(
        run_id=state.run_id,
        status="completed" if deployed else ("evaluation_completed" if succeeded else "failed"),
        endpoint=deployment.service_url if deployed else None,
        deployment=deployment if deployed else None,
        model_card_markdown=(
            "# Autonomous ML pipeline result\n\n"
            f"- Terminal state: `{state.status}`\n"
            f"- Retry count: `{state.retry_count}`\n"
        ),
        lineage=lineage,
        trace_url=langfuse_trace_url(state.run_id),
        evaluation=state.evaluation_result,
        runtime_seconds=runtime_seconds,
        error=error,
        llm_summary=summary,
        input_tokens=usage_report["input_tokens"],
        output_tokens=usage_report["output_tokens"],
        estimated_cost_usd=usage_report["estimated_cost_usd"],
        usage_report=usage_report,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the autonomous ML pipeline")
    parser.add_argument("--csv", required=True, help="Path to the input CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--goal", required=True, help="Prediction goal description")
    parser.add_argument(
        "--force-fail-eval",
        action="store_true",
        help="Set the evaluation threshold to 0.99 to demonstrate retries",
    )
    return parser


def _configure_demo_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in ("mcp", "orchestrator.flow"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _quiet_callbacks() -> FlowCallbacks:
    def ignore(_: Any) -> None:
        return None

    return FlowCallbacks(
        on_stage_start=ignore,
        on_stage_complete=ignore,
        on_stage_error=ignore,
        on_retry=ignore,
        on_validation_failure=ignore,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate inputs and run the existing CrewAI Flow as a console demo."""

    _configure_demo_output()
    args = _parser().parse_args(argv)
    server: _A2AServer | None = None
    try:
        task_type = infer_task_type(args.csv, args.target)
        state = create_pipeline_state(
            csv_path=args.csv,
            target_column=args.target,
            task_type=task_type,
            prediction_goal=args.goal,
        )
        artifact_dir = Path("artifacts") / "demo" / state.run_id.hex
        tracking_database = (artifact_dir / "mlflow.db").resolve()
        progress = _ProgressExporter(force_fail_eval=args.force_fail_eval)
        get_tracer_provider().add_span_processor(SimpleSpanProcessor(progress))
        server = _start_a2a_server(artifact_dir)
        flow = build_flow(
            state,
            callbacks=_quiet_callbacks(),
            a2a_url=DEFAULT_A2A_URL,
            tracking_uri=f"sqlite:///{tracking_database.as_posix()}",
            artifact_dir=artifact_dir,
            evaluation_threshold=(
                (0.99 if str(state.task_type) == "classification" else 0.0)
                if args.force_fail_eval
                else None
            ),
            suppress_flow_events=True,
        )
        started = time.perf_counter()
        flow.kickoff()
        runtime_seconds = time.perf_counter() - started
        build_llm_safe_context(flow.state)
        result = _pipeline_result(
            flow.state,
            runtime_seconds,
            llm_summary=flow.llm_summary,
        )
    except (DataPrepError, GuardrailViolation, ValueError) as exc:
        print(f"[PIPELINE]    ✗ FAILED_INPUT — {exc}", flush=True)
        flush_traces()
        return 2
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"[PIPELINE]    ✗ FAILED_INPUT — {exc}", flush=True)
        flush_traces()
        return 2
    finally:
        if server is not None:
            server.stop()

    if flow.state.status == PipelineStatus.SUCCESS:
        print(
            f"[PIPELINE]    ✓ SUCCESS — run_id: {result.run_id}",
            flush=True,
        )
    elif flow.state.status == PipelineStatus.FAILED_EVAL_EXHAUSTED:
        print(
            "[PIPELINE]    ✗ FAILED_EVAL_EXHAUSTED "
            f"after {flow.state.retry_count} retries",
            flush=True,
        )
    else:
        print(f"[PIPELINE]    ✗ {flow.state.status} — {result.error}", flush=True)
    report = result.usage_report
    print(
        f"[USAGE]       runtime {result.runtime_seconds:.1f}s | "
        f"{report.get('llm_requests', 0)} LLM calls "
        f"({report.get('local_llm_requests', 0)} local), "
        f"{report.get('cache_hits', 0)} cache hits, "
        f"{report.get('llm_latency_seconds', 0):.1f}s in LLMs | "
        f"tokens in/out {result.input_tokens}/{result.output_tokens} | "
        f"est. cost ${result.estimated_cost_usd:.6f}",
        flush=True,
    )
    for agent, bucket in report.get("by_agent", {}).items():
        print(
            f"              - {agent}: {bucket['requests']} calls, "
            f"{bucket.get('cache_hits', 0)} cached, "
            f"{bucket['total_tokens']} tokens, ${bucket['estimated_cost_usd']:.6f}",
            flush=True,
        )
    if result.deployment is not None:
        print(
            f"[DEPLOY]      {result.deployment.health_status} · {result.endpoint} · "
            f"{result.deployment.deployment_revision}",
            flush=True,
        )
    elif flow.deployment_error:
        print(f"[DEPLOY]      skipped — {flow.deployment_error}", flush=True)
    result_path = artifact_dir / "pipeline_result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    try:
        from orchestrator.insights import build_run_insights

        selection = flow.state.model_selection_result
        evaluation = flow.state.evaluation_result
        cli_final = {
            "run_id": str(result.run_id),
            "status": str(flow.state.status),
            "model_name": selection.best_model_name if selection else None,
            "metric_name": evaluation.primary_metric if evaluation else None,
            "metric_value": evaluation.held_out_metrics.get(evaluation.primary_metric) if evaluation else None,
            "threshold": evaluation.acceptance_threshold if evaluation else None,
            "retry_count": flow.state.retry_count,
            "runtime_ms": int(result.runtime_seconds * 1000),
            "estimated_cost_usd": result.estimated_cost_usd,
            "usage_report": result.usage_report,
            "trace_url": str(result.trace_url),
            "prediction_goal": flow.state.prediction_goal,
        }
        insights = build_run_insights(
            state=flow.state,
            attempts=flow.attempts,
            tracking_uri=flow.tracking_uri,
            artifact_dir=artifact_dir,
            final=cli_final,
        )
        (artifact_dir / "model_card.md").write_text(insights["model_card_markdown"], encoding="utf-8")
        for note in insights.get("findings", []):
            print(f"[INSIGHT]     {note}", flush=True)
    except Exception as exc:  # insights are a bonus; the run already succeeded
        print(f"[INSIGHT]     unavailable ({type(exc).__name__})", flush=True)
    print(f"[RESULT]      {result_path}", flush=True)
    print(f"[TRACE]       {result.trace_url}", flush=True)
    flush_traces()
    return 0 if flow.state.status == PipelineStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
