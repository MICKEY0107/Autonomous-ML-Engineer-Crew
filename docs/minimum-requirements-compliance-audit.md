# Minimum Requirements — Evidence Map

**Project:** Autonomous ML Engineer Crew (CSE3101 Agentic AI, BML Munjal University)
**Team:** Punya Mahajan (230611) · Aniket Mishra (230648) · Radhika Singh (230833)
**Updated:** 14 November 2026 (114 automated tests passing; native Google ADK and application-level evaluations are separated honestly.)

Requirements 1–10 are implemented and locally verified. Requirement 11 is container-ready but needs the
user's Google Cloud project for the final public deployment. Run the tests with
`.venv\Scripts\python.exe -m pytest -q` (114 passed). Native Google ADK evaluation is run with
`.venv\Scripts\python.exe scripts\run_adk_native_eval.py`; the six application-level ADK/MCP cases are run with
`.venv\Scripts\python.exe tests\adk_eval\run_eval.py`.

| # | Requirement | Where it is implemented | Tests that prove it | Where to see it live |
|---|---|---|---|---|
| 1 | Context sharing and state updates | `contracts/models.py` (`PipelineState`), `orchestrator/flow.py` (every stage reads and writes the shared state) | `test_contracts.py::test_pipeline_state_rejects_exhausted_retry_budget`, `::test_completed_state_requires_all_successful_stage_results`; `test_flow.py::test_flow_sends_feedback_and_succeeds_on_retry` | **Trace & contracts** tab; **Models** tab shows attempt-to-attempt feedback |
| 2 | At least two external tools through MCP | `tools/profile_dataset.py` and `tools/log_experiment.py` (FastMCP servers), called from `agents/data_prep.py` and `agents/model_selection_adk.py` | `test_profile_dataset.py` (3), `test_log_experiment.py` (3), `test_guardrails.py::test_profile_mcp_result_is_strictly_validated`, `::test_experiment_mcp_result_is_strictly_validated` | Stage list shows **MCP: profile_dataset** and **MCP: log_experiment**; MLflow at <http://127.0.0.1:5055> |
| 3 | Pydantic structured final output | `contracts/models.py` (`DataPrepResult`, `ModelSelectionResult`, `EvaluationResult`, `DeploymentResult`, `PipelineResult`); CrewAI tasks use `output_pydantic` + a guardrail (`orchestrator/crew_agents.py`) | `test_contracts.py` (14 tests, incl. `test_all_six_models_publish_closed_json_schemas`, `test_extra_fields_fail_closed`) | **Result JSON** download; **Trace & contracts** tab |
| 4 | Suitable agentic design patterns | Sequential flow with a router, tool use, evaluator-optimizer **reflection loop** with bounded retries (max 2), planning (LLM plans data prep, model search, serving policy) — `orchestrator/flow.py`, `orchestrator/agentic_policy.py` | `test_flow.py::test_flow_reaches_success_without_retry`, `::test_flow_sends_feedback_and_succeeds_on_retry`, `::test_flow_stops_after_two_failed_retries` | **Retry-loop demo** switch → 3 attempts with feedback, ends `FAILED_EVAL_EXHAUSTED` |
| 5 | Monitoring and logging with callbacks | Flow callbacks (`on_stage_start/complete/error`, `on_retry`, `on_validation_failure`); Google ADK **agent, model and tool** callbacks (`agents/model_selection_adk.py`); live `/predict` monitoring (`orchestrator/serving_monitor.py`) | `test_flow.py::test_all_nine_callback_invocations_are_correlated_by_run_id`; `test_model_selection_adk.py::test_adk_callbacks_validate_task_and_record_duration`, `::test_model_and_tool_callbacks_fire_for_every_decision_and_mcp_call`, `::test_callbacks_block_rows_unknown_actions_and_bad_metrics`; `test_deployment.py::test_predict_calls_are_monitored` | Live stage telemetry; **Predict** tab → *Live monitoring* |
| 6 | A2A protocol between heterogeneous agents | CrewAI Flow → Google ADK agent over A2A: server via `to_a2a()` with Agent Card (`agents/model_selection_adk.py`), client via `a2a-sdk` (`orchestrator/a2a_client.py`) | `test_day2_integration.py::test_adk_agent_a2a` (real server, Agent Card discovery, `SendMessage`), `test_model_selection_adk.py::test_a2a_service_contract_is_strict_and_bound_to_localhost`, `test_tracing.py::test_a2a_span_has_task_retry_attributes_and_propagates_traceparent` | **Agents & A2A** tab: Agent Card + every request/response; card at <http://127.0.0.1:8080/.well-known/agent-card.json> |
| 7 | Both CrewAI and ADK, no other agentic framework | CrewAI: Flow + Data-Prep, Evaluation and Deployment agents (`orchestrator/crew_agents.py`); ADK: Model-Selection agent. Dependencies in `pyproject.toml` / `uv.lock` | `test_data_prep.py`, `test_evaluation.py`, `test_model_selection_adk.py`, `test_day2_integration.py` | **Agents & A2A** → *Who decided what* (framework per decision) |
| 8 | Guardrails and callback-based validation | Input guardrails: CSV size/rows/path traversal, prompt injection, toxic language, target checks; raw rows never reach an LLM; CrewAI task guardrails; ADK before-agent/model/tool callbacks fail closed; quality gate before deploy; serving-policy request limits | `test_guardrails.py` (9), `test_review_fixes.py`, `test_deployment.py::test_serving_policy_limits_are_enforced`, `::test_api_predict_endpoints_and_api_key` | **Safety demos** buttons (prompt injection, toxic goal, bad CSV) |
| 9 | Observability: tracing and provenance | Langfuse via OpenTelemetry, one trace per run across Flow → A2A → ADK (`traceparent` propagated); MLflow tracking + model registry; lineage in `PipelineResult`; SHA-256 dataset and bundle digests | `test_tracing.py` (6 tests, incl. redaction of row data) | **Langfuse trace** and **MLflow** buttons on every result |
| 10 | End-to-end performance, cost and runtime evaluation | Usage ledger per decision (tokens, cost, latency, source) in `orchestrator/usage.py`; benchmark `scripts/benchmark.py` (success paths ×3 datasets, failure paths, monthly cost scenario) | `test_efficiency.py` (7) | `reports/benchmark.md`; **Agents & A2A** cost table on every run. A final cloud run is still required for cloud latency/cost evidence. |
| 11 | Deployment | Root `Dockerfile` packages the dashboard/API, CrewAI Flow, MCP tools and internal ADK A2A service. `scripts/deploy_gcp.ps1` deploys it to Cloud Run using Secret Manager. Per-model deployment code remains in `agents/deployment.py`. | `test_deployment.py` (11), `test_orchestrator_api.py::test_dashboard_root_serves_compiled_frontend`; local Docker smoke test passed | Pending live Cloud Run URL after Google Cloud setup. |

## Honest limits

- **Cloud Run** needs a Google Cloud project with billing, the `gcloud` CLI, and four Secret Manager values
  (Gemini key, Langfuse public key, Langfuse secret key and Langfuse project ID). The complete production image
  has passed a local container health and dashboard smoke test; public deployment and cloud-only evaluation remain.
- The LLM decisions use Gemini first, then a local Ollama model, then OpenRouter, then a typed rule-based
  fallback. Every decision is labelled with its real source, so a fallback is never presented as an LLM answer.
- Only the Model-Selection agent is behind A2A; the other agents run inside the CrewAI orchestrator. This is
  the CrewAI ↔ ADK interoperability boundary the synopsis defines.
