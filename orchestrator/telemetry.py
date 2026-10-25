"""Shared OpenTelemetry/Langfuse tracing with row-data redaction."""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from dotenv import load_dotenv
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import Span, SpanKind, Status, StatusCode


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


INSTRUMENTATION_SCOPE = "autonomous_ml_engineer_crew"
SERVICE_NAME = "autonomous-ml-engineer-crew"
LANGFUSE_TRACE_ORIGIN = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
LANGFUSE_PROJECT_ID = os.environ["LANGFUSE_PROJECT_ID"]
REDACTED = "[REDACTED_RAW_ROW_DATA]"
_MAX_ATTRIBUTE_CHARS = 4_096
_SPAN_IO_UNSET = object()
LANGFUSE_OBSERVATION_INPUT = "langfuse.observation.input"
LANGFUSE_OBSERVATION_OUTPUT = "langfuse.observation.output"
_RAW_FIELD_NAMES = frozenset(
    {
        "dataframe",
        "dataset_rows",
        "raw_data",
        "raw_row",
        "raw_rows",
        "record",
        "records",
        "row",
        "rows",
        "sample_rows",
        "samples",
        "test_data",
        "training_data",
    }
)
_RAW_TEXT_PATTERN = re.compile(
    r"(?i)(?P<prefix>['\"]?(?:raw[_ -]?rows?|dataset[_ -]?rows|records?|"
    r"sample[_ -]?rows?|training[_ -]?data|test[_ -]?data)['\"]?\s*[:=]\s*)"
    r"(?P<value>\[[^\r\n]*|\{[^\r\n]*|[^,;\r\n]+)"
)
_requested_trace_id: ContextVar[int | None] = ContextVar(
    "autonomous_ml_requested_trace_id",
    default=None,
)


class _RunIdGenerator(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        requested = _requested_trace_id.get()
        return requested if requested else super().generate_trace_id()


_provider = TracerProvider(
    resource=Resource.create({"service.name": SERVICE_NAME}),
    id_generator=_RunIdGenerator(),
)
_tracer = _provider.get_tracer(INSTRUMENTATION_SCOPE)
_langfuse_client: Any | None = None
_langfuse_initialized = False
_langfuse_lock = threading.Lock()


def langfuse_trace_url(run_id: UUID) -> str:
    """Return the project-scoped Langfuse URL for a pipeline state's run ID."""

    return (
        f"{LANGFUSE_TRACE_ORIGIN}/project/{LANGFUSE_PROJECT_ID}"
        f"/traces/{run_id.hex}"
    )


def _normalized_field_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")


def _is_raw_field(name: Any) -> bool:
    normalized = _normalized_field_name(name)
    if normalized in _RAW_FIELD_NAMES:
        return True
    parts = set(normalized.split("_"))
    return "raw" in parts and bool(parts & {"data", "record", "records", "row", "rows"})


def redact_text(value: str) -> str:
    """Redact serialized raw-row fields from otherwise safe error/log text."""

    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return _RAW_TEXT_PATTERN.sub(lambda match: match.group("prefix") + REDACTED, value)
    if isinstance(decoded, (Mapping, list, tuple)):
        return json.dumps(redact_payload(decoded), sort_keys=True, default=str)
    return value


def redact_payload(value: Any) -> Any:
    """Return a telemetry-safe copy with row-shaped values removed."""

    if isinstance(value, pd.DataFrame):
        return REDACTED
    if isinstance(value, BaseException):
        return redact_text(str(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            safe_key = str(key)
            redacted[safe_key] = REDACTED if _is_raw_field(key) else redact_payload(child)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if any(isinstance(item, Mapping) for item in value):
            return REDACTED
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def telemetry_attributes(fields: Mapping[str, Any]) -> dict[str, str | bool | int | float]:
    """Convert arbitrary fields to bounded OpenTelemetry scalar attributes."""

    attributes: dict[str, str | bool | int | float] = {}
    for key, value in fields.items():
        if value is None:
            continue
        safe_value = REDACTED if _is_raw_field(key) else redact_payload(value)
        if isinstance(safe_value, (str, bool, int, float)):
            rendered: str | bool | int | float = safe_value
        else:
            rendered = json.dumps(safe_value, sort_keys=True, default=str)
        if isinstance(rendered, str):
            rendered = rendered[:_MAX_ATTRIBUTE_CHARS]
        attributes[str(key)] = rendered
    return attributes


def _mask_langfuse_spans(*, params: Any) -> Any:
    """Apply a final redaction pass immediately before Langfuse export."""

    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches: dict[Any, Any] = {}
    for identifier, span in params.spans.items():
        replacements: dict[str, str | bool | int | float] = {}
        for key, value in span.attributes.items():
            if _is_raw_field(key):
                replacements[key] = REDACTED
            elif isinstance(value, str):
                safe_value = redact_text(value)
                if safe_value != value:
                    replacements[key] = safe_value[:_MAX_ATTRIBUTE_CHARS]
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


def configure_langfuse() -> Any | None:
    """Attach Langfuse to the shared provider when credentials are configured."""

    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client
    with _langfuse_lock:
        if _langfuse_initialized:
            return _langfuse_client
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if public_key and secret_key:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=os.getenv("LANGFUSE_BASE_URL"),
                tracer_provider=_provider,
                should_export_span=lambda span: (
                    span.instrumentation_scope.name == INSTRUMENTATION_SCOPE
                ),
                mask=lambda *, data, **_: redact_payload(data),
                mask_otel_spans=_mask_langfuse_spans,
            )
        _langfuse_initialized = True
    return _langfuse_client


@contextmanager
def traced_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, Any] | None = None,
    context: Context | None = None,
) -> Iterator[Span]:
    """Create a sanitized span that never records exception payloads verbatim."""

    configure_langfuse()
    with _tracer.start_as_current_span(
        name,
        context=context,
        kind=kind,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        if attributes:
            span.set_attributes(telemetry_attributes(attributes))
        try:
            yield span
        except BaseException as exc:
            safe_message = redact_text(str(exc))[:_MAX_ATTRIBUTE_CHARS]
            span.set_attributes(
                telemetry_attributes(
                    {
                        "error.type": type(exc).__name__,
                        "error.message": safe_message,
                    }
                )
            )
            span.set_status(Status(StatusCode.ERROR, safe_message))
            raise


@contextmanager
def pipeline_trace(run_id: UUID, **attributes: Any) -> Iterator[Span]:
    """Start a root trace whose 128-bit trace ID is exactly ``run_id``."""

    token = _requested_trace_id.set(run_id.int)
    try:
        with traced_span(
            "pipeline.run",
            context=Context(),
            attributes={"ml.run_id": str(run_id), **attributes},
        ) as span:
            yield span
    finally:
        _requested_trace_id.reset(token)


def inject_trace_headers(headers: dict[str, str]) -> None:
    inject(headers)


@contextmanager
def extracted_trace_context(carrier: Mapping[str, str]) -> Iterator[None]:
    token = otel_context.attach(extract(carrier))
    try:
        yield
    finally:
        otel_context.detach(token)


def add_trace_event(name: str, fields: Mapping[str, Any]) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, telemetry_attributes(fields))


def set_current_span_attributes(fields: Mapping[str, Any]) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(telemetry_attributes(fields))


def update_span_io(
    span: Span,
    *,
    input: Any = _SPAN_IO_UNSET,
    output: Any = _SPAN_IO_UNSET,
) -> None:
    """Set sanitized Langfuse I/O on an OpenTelemetry observation span.

    Langfuse reads these canonical attributes when ingesting raw OpenTelemetry
    spans. Every payload passes through the row-data redactor here and again in
    the export mask, so callers cannot accidentally attach dataset records.
    """

    if not span.is_recording():
        return
    attributes: dict[str, str] = {}
    if input is not _SPAN_IO_UNSET:
        attributes[LANGFUSE_OBSERVATION_INPUT] = json.dumps(
            redact_payload(input), sort_keys=True, default=str
        )
    if output is not _SPAN_IO_UNSET:
        attributes[LANGFUSE_OBSERVATION_OUTPUT] = json.dumps(
            redact_payload(output), sort_keys=True, default=str
        )
    if attributes:
        span.set_attributes(attributes)


def get_tracer_provider() -> TracerProvider:
    """Expose the provider for application shutdown and test exporters."""

    return _provider


def flush_traces() -> None:
    if _langfuse_client is not None:
        _langfuse_client.flush()
    else:
        _provider.force_flush()
