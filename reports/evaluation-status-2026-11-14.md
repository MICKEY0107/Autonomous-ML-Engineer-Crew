# Evaluation Status — 14 November 2026

## Verified locally

| Layer | Evidence | Result |
|---|---|---|
| Repository test suite | `python -m pytest -q` | 114/114 passed |
| Native Google ADK evaluation | `python scripts/run_adk_native_eval.py` | 2/2 guardrail cases passed; `response_match_score=1.0`, `tool_trajectory_avg_score=1.0` |
| Application-level ADK/MCP evaluation | `python tests/adk_eval/run_eval.py` | 6/6 passed |
| Production container | Root `Dockerfile`, Docker smoke test | Builds; `/api/health` returns `status: ok`; dashboard and sample API return HTTP 200 |

## What each evaluation proves

- Native ADK: actual ADK `AgentEvaluator` invokes the `root_agent` and verifies malformed-request and prompt-injection rejection. Its CSV is `reports/adk_native_eval.csv`.
- Application evaluation: trains three candidate models, records MLflow MCP actions, checks callback events, validates typed results, exercises retry handling, and verifies the injection callback blocks before training. Its reports are `adk_eval_report.json`, `adk_eval_report.md`, and `eval_results_report.html`.
- Full suite: contracts, CrewAI shared state flow, A2A client/server integration, MCP tools, guardrails, callbacks, deployment policy, tracing and API behavior.

## Still required after Google Cloud setup

1. Create the Secret Manager values required by `scripts/deploy_gcp.ps1`.
2. Deploy the complete image to Cloud Run.
3. Run a live pipeline through the public HTTPS URL and retain its Langfuse trace link.
4. Record cloud latency, LLM token/cost ledger, A2A/MCP events, structured result and response accuracy for the final report.

Cloud-only evidence will be added only after the public Cloud Run URL exists; local Docker results are not presented as a public-cloud deployment.
