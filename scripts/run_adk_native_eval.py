"""Run Google ADK's native evaluator against the model-selection agent."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_MODE", "offline")
os.environ.setdefault("LLM_CACHE", "off")
os.environ.setdefault("DEPLOY_ON_SUCCESS", "off")

print("Loading Google ADK evaluator...", flush=True)
from google.adk.evaluation.agent_evaluator import AgentEvaluator
print("Google ADK evaluator loaded.", flush=True)


EVAL_FILE = PROJECT_ROOT / "tests" / "adk_native_eval" / "guardrails.test.json"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "adk_native_eval.csv"


def main() -> int:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {EVAL_FILE}...", flush=True)
    asyncio.run(
        AgentEvaluator.evaluate(
            agent_module="adk_model_selection",
            eval_dataset_file_path_or_dir=str(EVAL_FILE),
            num_runs=1,
            print_detailed_results=True,
            output_file=str(OUTPUT_FILE),
        )
    )
    print(f"Native ADK evaluation passed; evidence: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
