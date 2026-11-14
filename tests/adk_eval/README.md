# ADK Evaluation Set — Model Selection Agent

This directory contains the **Google ADK-format evaluation set** for the
`ModelSelectionAgent` (defined in `agents/model_selection_adk.py`).
It follows the faculty demo format from:
<https://github.com/soharabhossain/Agentic_AI/tree/main/DEMO/ADK/Eval>

---

## What This Eval Set Tests

The evaluation suite validates the **Google ADK Model Selection Agent's A2A interface**
across four evaluation layers:

### Layer 1 — Contract Evaluation
Pydantic schema validation, field presence, and type constraints for both
`ModelSelectionRequest` (inbound) and `ModelSelectionResult` (outbound).
Cases `bad_input_rejected` and `injection_blocked` exercise fail-closed rejection.

### Layer 2 — Agent & Protocol Evaluation
A2A Agent Card availability, `before_agent_callback` / `after_agent_callback` firing,
MCP `SendMessage` routing, and the five allowlisted MCP tool actions:
`create_run`, `log_params`, `log_metrics`, `query_best_run`, `register_model`.

### Layer 3 — ML Evaluation
Held-out candidate metrics, correct metric direction (`f1_weighted` for classification,
`rmse` for regression), candidate comparison logic, and retry behavior with
`retry_attempt` / `retry_feedback` fields.

### Layer 4 — System Evaluation
p50/p95 runtime per candidate, token counts (via `llm_usage`), cost estimation,
and deployment-health checks after model registration.

---

## Files

| File | Purpose |
|------|---------|
| `evalset.json` | 6 ADK eval cases (the official `.evalset.json` format) |
| `eval_config.json` | Metric threshold configuration |
| `run_eval.py` | Standalone evaluation script (offline mode, produces JSON + MD reports) |
| `README.md` | This file |

---

## Eval Cases

| ID | Task | What It Verifies |
|----|------|-----------------|
| `classification_success` | Iris-style 3-class classification | All 3 candidates trained, f1_weighted/accuracy logged, model registered |
| `regression_success` | House prices regression | RMSE+MAE logged, Ridge/LinearReg selected, model registered |
| `retry_with_feedback` | Regression retry (attempt=1) | retry_attempt=1 in result, new hyperparameters reflecting feedback |
| `bad_input_rejected` | Malformed payload (missing `data_prep_result`) | `before_agent_callback` rejects with `{"error": "rejected: ..."}` |
| `injection_blocked` | `prediction_goal` with injection text, file URIs missing | Artifact-not-found guard blocks training |
| `classification_all_metrics` | Credit-approval classification | All 5 MCP actions appear in CALLBACK_EVENTS, full `candidates` list in result |

---

## How to Run

### Using the ADK CLI (requires `adk` installed)

```bash
adk eval agents/model_selection_adk.py tests/adk_eval/evalset.json
```

### Using the standalone eval script (offline, no LLM required)

```bash
.venv\Scripts\python.exe tests\adk_eval\run_eval.py
```

Reports are written to:
- `reports/adk_eval_report.json` — machine-readable per-case evidence
- `reports/adk_eval_report.md` — human-readable pass/fail table

### Using pytest

```bash
.venv\Scripts\python.exe -m pytest tests/test_adk_eval.py -v
```

---

## Reference

- Faculty demo: <https://github.com/soharabhossain/Agentic_AI/tree/main/DEMO/ADK/Eval>
- ADK Agent evaluation docs: <https://google.github.io/adk-docs/evaluate/>
- Project agent: `agents/model_selection_adk.py`
- Request/response contracts: `contracts/models.py`
