"""CrewAI Flow orchestration for the autonomous ML pipeline.

Flow exports are loaded lazily so low-level telemetry can be imported by agents
without importing the Flow (and therefore every agent) recursively.
"""

from typing import Any

__all__ = [
    "AutonomousMLEngineerFlow",
    "FlowCallbackEvent",
    "FlowCallbacks",
    "build_flow",
    "run_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import flow

        return getattr(flow, name)
    raise AttributeError(name)
