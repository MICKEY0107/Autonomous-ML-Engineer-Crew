"""CrewAI Flow for data preparation, A2A selection, and evaluation.

The normal route is::

    data_prep -> model_selection (A2A) -> evaluation -> SUCCESS

Failed evaluation feeds ``retry_feedback`` back to the ADK service for at most
two retuning attempts. The flow terminates with one of three explicit status
values: ``SUCCESS``, ``FAILED_EVAL_EXHAUSTED``, or ``FAILED_INPUT``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import UUID, uuid4

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from opentelemetry.trace import SpanKind

from agents.data_prep import run_data_prep
from agents.evaluation import evaluate_model
from agents.model_selection_adk import ModelSelectionRequest
from orchestrator import usage
from orchestrator.a2a_client import agent_base_url, send_model_selection_task
from orchestrator.a2a_client import reset_log as a2a_log_reset
from contracts.models import (
    DataPrepResult,
    DeploymentResult,
    EvaluationResult,
    ModelSelectionResult,
    PipelineStage,
    PipelineState,
    PipelineStatus,
    TaskType,
)
from orchestrator.telemetry import (
    add_trace_event,
    inject_trace_headers,
    pipeline_trace,
    redact_payload,
    redact_text,
    set_current_span_attributes,
    traced_span,
    update_span_io,
)
from orchestrator.llm_reasoning import generate_pipeline_summary


logger = logging.getLogger(__name__)
MAX_EVALUATION_RETRIES = 2
# Base URL of the ADK agent; its Agent Card is discovered at
# ``/.well-known/agent-card.json``. A2A_TASKS_URL is still read for older .env files.
DEFAULT_A2A_URL = agent_base_url(
    os.getenv("A2A_AGENT_URL") or os.environ["A2A_TASKS_URL"]
)
DEFAULT_ARTIFACT_DIR = Path("/tmp/artifacts")


class FlowCallbackEvent(BaseModel):
    """Sanitized lifecycle event delivered to orchestration callbacks."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    name: str
    stage: str
    run_id: UUID
    retry_count: int = Field(ge=0)
    status: PipelineStatus
    message: str | None = None


Callback = Callable[[FlowCallbackEvent], None]


def on_stage_start(event: FlowCallbackEvent) -> None:
    logger.info("stage started: %s", redact_payload(event.model_dump(mode="json")))


def on_stage_complete(event: FlowCallbackEvent) -> None:
    logger.info("stage completed: %s", redact_payload(event.model_dump(mode="json")))


def on_stage_error(event: FlowCallbackEvent) -> None:
    logger.error("stage failed: %s", redact_payload(event.model_dump(mode="json")))


def on_retry(event: FlowCallbackEvent) -> None:
    logger.warning("evaluation retry: %s", redact_payload(event.model_dump(mode="json")))


def on_validation_failure(event: FlowCallbackEvent) -> None:
    logger.error(
        "validation failed: %s", redact_payload(event.model_dump(mode="json"))
    )


class FlowCallbacks(BaseModel):
    """Replaceable callback collection with safe logging defaults."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    on_stage_start: Callback = on_stage_start
    on_stage_complete: Callback = on_stage_complete
    on_stage_error: Callback = on_stage_error
    on_retry: Callback = on_retry
    on_validation_failure: Callback = on_validation_failure


def _default_tracking_uri() -> str:
    database = (DEFAULT_ARTIFACT_DIR / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("dataset_uri must use the local file scheme")
    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    elif len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(url2pathname(path)).resolve()


def _safe_error(error: BaseException) -> str:
    message = " ".join(redact_text(str(error)).split())
    return f"{type(error).__name__}: {message}"[:500]


def _selection_metric(
    result: ModelSelectionResult,
    task_type: Any,
) -> tuple[str, float]:
    """Return the metric name paired with the contract's selected value."""

    primary = "f1_weighted" if str(task_type) == "classification" else "rmse"
    preferences = (primary, "accuracy", "roc_auc", "r2", "mae")
    for name in preferences:
        if name in result.training_metrics:
            return name, float(result.training_metrics[name])
    name, value = next(iter(result.training_metrics.items()))
    return name, float(value)


def _usage_source(records: list[dict[str, Any]]) -> str:
    """Summarise where the ADK agent's decisions came from (for honest labels)."""

    # One outcome per decision stage: the last record written for that stage.
    outcome: dict[str, str] = {}
    for record in records:
        outcome[str(record.get("stage", ""))] = str(record.get("source", ""))
    sources = set(outcome.values())
    if not sources or sources == {"fallback"}:
        return "fallback"
    if "fallback" in sources:
        return "partial_fallback"
    if sources == {"cache"}:
        return "cache"
    return "llm" if "llm" in sources else "secondary_llm"


class AutonomousMLEngineerFlow(Flow[PipelineState]):
    """Typed CrewAI orchestration with a bounded evaluator-optimizer loop."""

    callbacks: FlowCallbacks = Field(default_factory=FlowCallbacks)
    a2a_url: str = DEFAULT_A2A_URL
    tracking_uri: str = Field(default_factory=_default_tracking_uri)
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR)
    evaluation_threshold: float | None = None
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_summary: str = ""
    # One entry per model-selection/evaluation attempt, for the dashboard's
    # retry timeline (row-free: candidates, metrics, feedback).
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    deployment_error: str | None = None

    async def kickoff_async(self, *args: Any, **kwargs: Any) -> Any:
        """Run the complete Flow under a root trace keyed by ``run_id``."""

        self.state.trace_id = self.state.run_id.hex
        usage.reset()  # fresh token/cost ledger for this pipeline run
        a2a_log_reset()
        started = time.perf_counter()
        with pipeline_trace(
            self.state.run_id,
            **{
                "ml.task_type": self.state.task_type,
                "ml.target_column": self.state.target_column,
                "ml.dataset_uri": self.state.dataset_uri,
            },
        ) as span:
            update_span_io(
                span,
                input={
                    "csv_path": str(_local_path(self.state.dataset_uri)),
                    "target_column": self.state.target_column,
                    "prediction_goal": self.state.prediction_goal,
                },
            )
            try:
                result = await super().kickoff_async(*args, **kwargs)
                selection = self.state.model_selection_result
                evaluation = self.state.evaluation_result
                metric_name = (
                    evaluation.primary_metric if evaluation is not None else "metric"
                )
                metric_value = (
                    float(evaluation.held_out_metrics[metric_name])
                    if evaluation is not None
                    else 0.0
                )
                self.llm_summary = generate_pipeline_summary(
                    str(self.state.status),
                    selection.best_model_name if selection is not None else "unknown",
                    metric_name,
                    metric_value,
                    self.state.retry_count,
                    time.perf_counter() - started,
                )
                return result
            finally:
                update_span_io(
                    span,
                    output={
                        "status": str(self.state.status),
                        "run_id": str(self.state.run_id),
                    },
                )

    def _callback(
        self,
        name: Literal[
            "on_stage_start",
            "on_stage_complete",
            "on_stage_error",
            "on_retry",
            "on_validation_failure",
        ],
        stage: str,
        message: str | None = None,
    ) -> None:
        event = FlowCallbackEvent(
            name=name,
            stage=stage,
            run_id=self.state.run_id,
            retry_count=self.state.retry_count,
            status=self.state.status,
            message=message,
        )
        add_trace_event(name, event.model_dump(mode="json"))
        try:
            getattr(self.callbacks, name)(event)
        except Exception:
            logger.exception("flow callback %s failed", name)

    def _execute_stage(self, stage: str, operation: Callable[[], Any]) -> Any:
        with traced_span(
            f"agent.{stage}",
            attributes={
                "ml.agent": stage,
                "ml.run_id": self.state.run_id,
                "ml.task_type": self.state.task_type,
                "ml.retry_count": self.state.retry_count,
            },
        ) as span:
            self._callback("on_stage_start", stage)
            try:
                result = operation()
            except ValidationError as exc:
                message = _safe_error(exc)
                self._callback("on_validation_failure", stage, message)
                self._callback("on_stage_error", stage, message)
                raise
            except ValueError as exc:
                message = _safe_error(exc)
                self._callback("on_validation_failure", stage, message)
                self._callback("on_stage_error", stage, message)
                raise
            except Exception as exc:
                self._callback("on_stage_error", stage, _safe_error(exc))
                raise
            span_input, span_output = self._stage_span_io(stage, result)
            update_span_io(span, input=span_input, output=span_output)
            if isinstance(result, DataPrepResult):
                set_current_span_attributes(
                    {"ml.feature_count": len(result.feature_columns)}
                )
            elif isinstance(result, ModelSelectionResult):
                metric_name, _ = _selection_metric(result, self.state.task_type)
                set_current_span_attributes({"ml.metric_name": metric_name})
            elif isinstance(result, EvaluationResult):
                set_current_span_attributes(
                    {"ml.metric_name": result.primary_metric}
                )
            self._callback("on_stage_complete", stage)
            return result

    def _stage_span_io(
        self,
        stage: str,
        result: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the row-free Langfuse payload for a completed agent stage."""

        if stage == "data_prep" and isinstance(result, DataPrepResult):
            return (
                {
                    "csv_path": str(_local_path(self.state.dataset_uri)),
                    "target_column": self.state.target_column,
                },
                {
                    "train_path": result.train_path,
                    "test_path": result.test_path,
                    "row_count": result.row_count,
                },
            )
        if stage == "model_selection" and isinstance(result, ModelSelectionResult):
            _, metric_value = _selection_metric(result, self.state.task_type)
            return (
                {
                    "task_type": str(self.state.task_type),
                    "retry_count": self.state.retry_count,
                },
                {
                    "best_model_name": result.best_model_name,
                    "metric_value": metric_value,
                },
            )
        if stage == "deployment" and isinstance(result, DeploymentResult):
            return (
                {"model_uri": self.state.model_selection_result.best_model_uri
                 if self.state.model_selection_result else None},
                {
                    "service_url": str(result.service_url),
                    "health_status": str(result.health_status),
                    "revision": result.deployment_revision,
                    "digest": result.image_digest,
                },
            )
        if stage == "evaluation" and isinstance(result, EvaluationResult):
            metric_name = result.primary_metric
            metric_value = float(result.held_out_metrics[metric_name])
            return (
                {
                    "metric_value": metric_value,
                    "threshold": result.acceptance_threshold,
                },
                {
                    "passed": result.passed,
                    "feedback": result.feedback,
                },
            )
        raise ValueError(f"unsupported completed stage for tracing: {stage}")

    def _append_error(self, message: str) -> None:
        self.state.errors = [*self.state.errors, message[:500]]

    def _mark_failed_input(self, error: BaseException) -> None:
        self._append_error(_safe_error(error))
        self.state.current_stage = PipelineStage.COMPLETE
        self.state.status = PipelineStatus.FAILED_INPUT

    def _mark_failed_evaluation(self, message: str) -> None:
        self._append_error(message)
        self.state.current_stage = PipelineStage.COMPLETE
        self.state.status = PipelineStatus.FAILED_EVAL_EXHAUSTED

    def _mark_success(self) -> None:
        self._deploy()
        self.state.current_stage = PipelineStage.COMPLETE
        self.state.status = PipelineStatus.SUCCESS

    def _deploy(self) -> None:
        """Deployment and Observability Agent: package + serve the accepted model.

        A deployment problem never turns a passing model into a failed run; it
        is reported in ``deployment_error`` and the result stays evaluation-only.
        """

        if os.getenv("DEPLOY_ON_SUCCESS", "on").strip().casefold() in {"off", "0", "false", "no"}:
            return
        from agents.deployment import deploy_model  # heavy imports only when deploying

        self.state.current_stage = PipelineStage.DEPLOYMENT
        try:
            self.state.deployment_result = self._execute_stage(
                "deployment",
                lambda: deploy_model(self.state, tracking_uri=self.tracking_uri),
            )
        except Exception as exc:
            self.deployment_error = _safe_error(exc)

    def _model_request(self, retry_feedback: str | None = None) -> ModelSelectionRequest:
        if self.state.data_prep_result is None:
            raise ValueError("data preparation result is required for model selection")
        return ModelSelectionRequest(
            data_prep_result=self.state.data_prep_result,
            task_type=self.state.task_type,
            tracking_uri=self.tracking_uri,
            experiment_name="autonomous-model-selection",
            retry_feedback=retry_feedback,
            retry_attempt=self.state.retry_count,
            prediction_goal=self.state.prediction_goal,
        )

    def _post_model_selection(
        self,
        retry_feedback: str | None = None,
    ) -> ModelSelectionResult:
        request = self._model_request(retry_feedback)
        headers: dict[str, str] = {}
        with traced_span(
            "a2a.message.send",
            kind=SpanKind.CLIENT,
            attributes={
                "rpc.system": "a2a",
                "rpc.method": "SendMessage",
                "server.address": self.a2a_url,
                "ml.run_id": self.state.run_id,
                "ml.task_type": self.state.task_type,
                "ml.retry_count": self.state.retry_count,
            },
        ) as span:
            update_span_io(
                span,
                input={
                    "task_type": str(self.state.task_type),
                    "retry_count": self.state.retry_count,
                },
            )
            inject_trace_headers(headers)
            # Real A2A: discover the ADK Agent Card, then SendMessage (JSON-RPC).
            result, call_info = send_model_selection_task(
                self.a2a_url,
                request.model_dump(mode="json"),
                headers=headers,
                timeout_seconds=self.request_timeout_seconds,
            )
            usage.extend(result.llm_usage)  # ADK agent's LLM calls, returned over A2A
            set_current_span_attributes(
                {
                    "a2a.agent.name": call_info.agent_name,
                    "a2a.agent.url": call_info.agent_url,
                    "a2a.protocol_version": call_info.protocol_version,
                    "a2a.task.id": call_info.task_id,
                    "a2a.task.state": call_info.task_state,
                }
            )
            _, metric_value = _selection_metric(result, self.state.task_type)
            update_span_io(
                span,
                output={
                    "best_model_name": result.best_model_name,
                    "metric_value": metric_value,
                },
            )
            with traced_span(
                "llm.reasoning.model_selection_result",
                attributes={
                    "ml.run_id": self.state.run_id,
                    "ml.reasoning.source": "a2a_response",
                    "ml.reasoning.row_safe": True,
                },
            ) as reasoning_span:
                update_span_io(
                    reasoning_span,
                    input={
                        "best_model_name": result.best_model_name,
                        "metric_value": metric_value,
                    },
                    output={
                        "reasoning": result.llm_reasoning,
                        # The ADK agent reports its live LLM calls; none = fallback policy.
                        "fallback_used": _usage_source(result.llm_usage) == "fallback",
                        "source": _usage_source(result.llm_usage),
                    },
                )
            return result

    def _prepare_data(self) -> DataPrepResult:
        return run_data_prep(
            csv_path=_local_path(self.state.dataset_uri),
            target_column=self.state.target_column,
            run_id=self.state.run_id,
            task_type=self.state.task_type,
            artifact_dir=self.artifact_dir,
            prediction_goal=self.state.prediction_goal,
        )

    def _evaluate_selection(self) -> EvaluationResult:
        if self.state.model_selection_result is None:
            raise ValueError("model selection result is required for evaluation")
        return evaluate_model(
            self.state.model_selection_result,
            tracking_uri=self.tracking_uri,
            acceptance_threshold=self.evaluation_threshold,
            artifact_dir=Path(self.artifact_dir) / "evaluation-cache",
        )

    def _store_selection(self, result: ModelSelectionResult) -> None:
        self.state.model_selection_result = result
        self.attempts.append(
            {
                "attempt": len(self.attempts) + 1,
                "retry_feedback_in": self.state.evaluation_result.retry_feedback
                if self.state.evaluation_result is not None and self.attempts
                else None,
                "selected_model": result.best_model_name,
                "candidates": result.candidates,
                "reasoning": result.llm_reasoning,
                "evaluation": None,
            }
        )
        self.state.artifact_references = {
            **self.state.artifact_references,
            "model": result.best_model_uri,
        }

    def _record_evaluation(self, result: EvaluationResult) -> None:
        if self.attempts:
            self.attempts[-1]["evaluation"] = {
                "passed": result.passed,
                "primary_metric": result.primary_metric,
                "held_out_metrics": result.held_out_metrics,
                "threshold": result.acceptance_threshold,
                "direction": str(result.metric_direction),
                "feedback": result.retry_feedback,
                "reasoning": result.llm_reasoning,
            }

    @start()
    def data_prep_stage(self) -> DataPrepResult | None:
        self.state.status = PipelineStatus.RUNNING
        self.state.current_stage = PipelineStage.DATA_PREPARATION
        try:
            result = self._execute_stage("data_prep", self._prepare_data)
        except Exception as exc:
            self._mark_failed_input(exc)
            return None
        self.state.data_prep_result = result
        self.state.artifact_references = {
            **self.state.artifact_references,
            "train": result.train_artifact_uri,
            "test": result.test_artifact_uri,
        }
        return result

    @router(data_prep_stage)
    def route_prepared_input(
        self,
        result: DataPrepResult | None,
    ) -> Literal["input_ready", "failed_input"]:
        return "input_ready" if result is not None else "failed_input"

    @listen("input_ready")
    def model_selection_stage(self, _: Any = None) -> ModelSelectionResult | None:
        self.state.current_stage = PipelineStage.MODEL_SELECTION
        try:
            result = self._execute_stage(
                "model_selection",
                self._post_model_selection,
            )
        except Exception as exc:
            self._mark_failed_input(exc)
            return None
        self._store_selection(result)
        return result

    @listen(model_selection_stage)
    def evaluation_stage(
        self,
        selection: ModelSelectionResult | None,
    ) -> EvaluationResult | None:
        if selection is None:
            return None
        self.state.current_stage = PipelineStage.EVALUATION
        try:
            result = self._execute_stage("evaluation", self._evaluate_selection)
        except Exception as exc:
            self._mark_failed_evaluation(_safe_error(exc))
            return None
        self.state.evaluation_result = result
        self._record_evaluation(result)
        return result

    @router(evaluation_stage)
    def route_evaluation(
        self,
        result: EvaluationResult | None,
    ) -> Literal["success", "retry", "failed_eval_exhausted", "failed_input"]:
        if self.state.status == PipelineStatus.FAILED_INPUT:
            return "failed_input"
        if self.state.status == PipelineStatus.FAILED_EVAL_EXHAUSTED:
            return "failed_eval_exhausted"
        if result is not None and result.passed:
            return "success"
        retry_limit = min(self.state.max_retries, MAX_EVALUATION_RETRIES)
        if self.state.retry_count < retry_limit:
            return "retry"
        return "failed_eval_exhausted"

    @listen("success")
    def terminal_success(self, _: Any = None) -> PipelineState:
        self._mark_success()
        return self.state

    @listen("failed_input")
    def terminal_failed_input(self, _: Any = None) -> PipelineState:
        return self.state

    @listen("failed_eval_exhausted")
    def terminal_failed_evaluation(self, _: Any = None) -> PipelineState:
        if self.state.status != PipelineStatus.FAILED_EVAL_EXHAUSTED:
            self._mark_failed_evaluation(
                "Model did not meet the evaluation gate after two retry attempts."
            )
        return self.state

    @listen("retry")
    def retry_until_terminal(self, _: Any = None) -> PipelineState:
        retry_limit = min(self.state.max_retries, MAX_EVALUATION_RETRIES)
        while self.state.retry_count < retry_limit:
            evaluation = self.state.evaluation_result
            feedback = (
                evaluation.retry_feedback
                if evaluation is not None and evaluation.retry_feedback
                else "Improve the primary held-out metric before resubmission."
            )
            self.state.retry_count += 1
            self._callback("on_retry", "evaluation", feedback)

            self.state.current_stage = PipelineStage.MODEL_SELECTION
            try:
                selection = self._execute_stage(
                    "model_selection",
                    lambda: self._post_model_selection(feedback),
                )
            except Exception as exc:
                self._mark_failed_input(exc)
                return self.state
            self._store_selection(selection)

            self.state.current_stage = PipelineStage.EVALUATION
            try:
                evaluated = self._execute_stage(
                    "evaluation",
                    self._evaluate_selection,
                )
            except Exception as exc:
                self._mark_failed_evaluation(_safe_error(exc))
                return self.state
            self.state.evaluation_result = evaluated
            self._record_evaluation(evaluated)
            if evaluated.passed:
                self._mark_success()
                return self.state

        self._mark_failed_evaluation(
            "Model did not meet the evaluation gate after two retry attempts."
        )
        return self.state


def build_flow(
    state: PipelineState,
    *,
    callbacks: FlowCallbacks | None = None,
    a2a_url: str = DEFAULT_A2A_URL,
    tracking_uri: str | None = None,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    evaluation_threshold: float | None = None,
    suppress_flow_events: bool = False,
) -> AutonomousMLEngineerFlow:
    """Create a configured flow around an already validated PipelineState."""

    return AutonomousMLEngineerFlow(
        initial_state=state,
        callbacks=callbacks or FlowCallbacks(),
        a2a_url=a2a_url,
        tracking_uri=tracking_uri or _default_tracking_uri(),
        artifact_dir=str(artifact_dir),
        evaluation_threshold=evaluation_threshold,
        suppress_flow_events=suppress_flow_events,
    )


def _failed_input_state(
    *,
    csv_path: str | Path,
    target_column: str,
    error: BaseException,
) -> PipelineState:
    """Create a contract-valid terminal state for pre-flow validation errors."""

    run_id = uuid4()
    try:
        dataset_uri = Path(csv_path).expanduser().resolve().as_uri()
    except (OSError, TypeError, ValueError):
        dataset_uri = (Path.cwd() / "invalid-input.csv").resolve().as_uri()
    return PipelineState(
        run_id=run_id,
        dataset_uri=dataset_uri,
        task_type=TaskType.REGRESSION,
        target_column=str(target_column).strip() or "invalid_target",
        prediction_goal="rejected input",
        status=PipelineStatus.FAILED_INPUT,
        current_stage=PipelineStage.COMPLETE,
        trace_id=run_id.hex,
        errors=[_safe_error(error)],
    )


def run_pipeline(
    csv_path: str | Path,
    target_column: str,
    prediction_goal: str,
) -> PipelineState:
    """Validate external inputs, execute the flow, and return its final state.

    Input failures are represented as ``FAILED_INPUT`` states so callers can
    handle guardrail rejections without catching implementation exceptions.
    """

    try:
        # Imported lazily to avoid a module-import cycle: main imports build_flow.
        from main import create_pipeline_state, infer_task_type

        task_type = infer_task_type(csv_path, target_column)
        state = create_pipeline_state(
            csv_path=csv_path,
            target_column=target_column,
            task_type=task_type,
            prediction_goal=prediction_goal,
        )
    except Exception as exc:
        return _failed_input_state(
            csv_path=csv_path,
            target_column=target_column,
            error=exc,
        )

    flow = build_flow(state)
    flow.kickoff()
    return flow.state
