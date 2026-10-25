"""Check that each configured LLM provider answers (prints no secrets)."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from orchestrator.llm_provider import crewai_llm, fallback_crewai_llm  # noqa: E402


def _clean(text: str) -> str:
    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "GEMINI_API_KEY", "LANGFUSE_SECRET_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "***")
    return re.sub(r"\s+", " ", text)[:400]


def probe(label: str, factory) -> None:
    started = time.perf_counter()
    try:
        llm = factory()
        if llm is None:
            print(f"{label}: not configured")
            return
        reply = llm.call("Reply with exactly: OK")
        print(f"{label}: OK model={getattr(llm, 'model', '?')} reply={_clean(str(reply))[:40]!r} "
              f"in {time.perf_counter() - started:.1f}s")
    except Exception as exc:  # report, never raise
        print(f"{label}: FAILED {type(exc).__name__}: {_clean(str(exc))}")


print(f"LLM_PROVIDER={os.getenv('LLM_PROVIDER')} CREWAI_MODEL={os.getenv('CREWAI_MODEL')}")
print(f"OPENROUTER key set: {bool(os.getenv('OPENROUTER_API_KEY') or os.getenv('OPEN_ROUTER_API_KEY'))}; "
      f"GEMINI key set: {bool(os.getenv('GEMINI_API_KEY'))}")
probe("primary", crewai_llm)
probe("secondary", fallback_crewai_llm)
