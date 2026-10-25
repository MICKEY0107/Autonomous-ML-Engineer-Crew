"""A2A protocol client used by the CrewAI Flow to call the Google ADK agent.

The CrewAI side never imports the ADK agent. It only knows the agent's base
URL, and talks to it the standard A2A way:

1. Discovery  - fetch the public Agent Card from
   ``<base>/.well-known/agent-card.json`` (``A2ACardResolver``).
2. Messaging  - build an A2A ``Message`` (role=user) whose single text part is
   the JSON-serialised ``ModelSelectionRequest`` and send it with the A2A
   ``SendMessage`` JSON-RPC method (``a2a.client.create_client``).
3. Result     - the ADK ``A2aAgentExecutor`` returns a ``Task``; the agent's
   reply (a ``ModelSelectionResult`` JSON string) is read from the task's
   artifacts / status message and validated with Pydantic (fail closed).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_message_text, get_artifact_text, new_text_message
from a2a.types.a2a_pb2 import Role, SendMessageRequest, TaskState
from pydantic import ValidationError

from contracts.models import ModelSelectionResult

LEGACY_TASK_SUFFIX = "/a2a/tasks/send"


class A2AProtocolError(RuntimeError):
    """The remote A2A agent failed or returned an unusable response."""


@dataclass
class A2ACallInfo:
    """Protocol-level facts recorded on the tracing span (no row data)."""

    agent_name: str = ""
    agent_url: str = ""
    protocol_version: str = ""
    task_id: str = ""
    task_state: str = ""
    response_kinds: list[str] = field(default_factory=list)


def agent_base_url(url: str) -> str:
    """Accept either the agent base URL or the old ``/a2a/tasks/send`` URL."""

    url = url.strip().rstrip("/")
    if url.endswith(LEGACY_TASK_SUFFIX):
        url = url[: -len(LEGACY_TASK_SUFFIX)]
    return url


def _parse_result(texts: list[str]) -> ModelSelectionResult:
    # The agent's final answer is the last text that validates as the contract.
    last_error: Exception | None = None
    for text in reversed(texts):
        try:
            return ModelSelectionResult.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_error = exc
    raise A2AProtocolError(
        f"A2A response did not contain a valid ModelSelectionResult: {last_error}"
    )


async def send_model_selection_task_async(
    base_url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 300.0,
) -> tuple[ModelSelectionResult, A2ACallInfo]:
    """Discover the ADK agent card and send one A2A message to it."""

    base_url = agent_base_url(base_url)
    info = A2ACallInfo()
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(headers=headers or {}, timeout=timeout) as http:
        card = await A2ACardResolver(http, base_url).get_agent_card()
        info.agent_name = card.name
        interfaces = list(getattr(card, "supported_interfaces", []) or [])
        if interfaces:
            info.agent_url = interfaces[0].url
            info.protocol_version = getattr(interfaces[0], "protocol_version", "")

        client = await create_client(
            card,
            client_config=ClientConfig(streaming=False, httpx_client=http),
        )
        message = new_text_message(
            json.dumps(payload),
            role=Role.ROLE_USER,
        )
        texts: list[str] = []
        try:
            async for response in client.send_message(SendMessageRequest(message=message)):
                if response.HasField("task"):
                    task = response.task
                    info.response_kinds.append("task")
                    info.task_id = task.id
                    info.task_state = TaskState.Name(task.status.state)
                    texts.extend(get_artifact_text(a) for a in task.artifacts)
                    if task.status.HasField("message"):
                        texts.append(get_message_text(task.status.message))
                    if task.status.state in (
                        TaskState.TASK_STATE_FAILED,
                        TaskState.TASK_STATE_REJECTED,
                        TaskState.TASK_STATE_CANCELED,
                    ):
                        detail = (
                            get_message_text(task.status.message)
                            if task.status.HasField("message")
                            else info.task_state
                        )
                        raise A2AProtocolError(f"ADK agent task {info.task_state}: {detail}")
                elif response.HasField("message"):
                    info.response_kinds.append("message")
                    texts.append(get_message_text(response.message))
        finally:
            await client.close()

    return _parse_result([t for t in texts if t]), info


_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="a2a-client")


def send_model_selection_task(
    base_url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 300.0,
) -> tuple[ModelSelectionResult, A2ACallInfo]:
    """Synchronous wrapper; safe to call from inside a running event loop."""

    context = contextvars.copy_context()
    future = _EXECUTOR.submit(
        context.run,
        asyncio.run,
        send_model_selection_task_async(
            base_url,
            payload,
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
    )
    return future.result(timeout=timeout_seconds + 30)
