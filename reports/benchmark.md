# End-to-end benchmark

Generated 2026-11-10 14:40 UTC · 1 runs per dataset · LLM_MODE=offline · LLM_CACHE=on · primary model `?` · fallback chain `none`

Each run is a full `main.py` execution: guardrails, CrewAI Data-Prep agent, MCP profiling, A2A call to the Google ADK model-selection agent, MLflow logging through MCP, CrewAI evaluation, bounded retry loop and the Deployment agent.

## Success paths

| Scenario | Runs | Passed | Runtime p50 | Runtime p95 | Mean LLM calls | Local / cached / rule-based | Mean tokens | Mean cost / run | Mean held-out metric | Deployed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| house prices (regression) | 1 | 1/1 | 4.2 s | 4.2 s | 0.0 | 0.0 / 0.0 / 5.0 | 0 | $0.00000 | rmse 18,271.4424 | 1/1 |
| customer churn (classification) | 1 | 1/1 | 4.4 s | 4.4 s | 0.0 | 0.0 / 0.0 / 5.0 | 0 | $0.00000 | f1_weighted 0.898 | 1/1 |
| iris (3-class) | 1 | 1/1 | 3.9 s | 3.9 s | 0.0 | 0.0 / 0.0 / 5.0 | 0 | $0.00000 | f1_weighted 0.9333 | 1/1 |

Across all 3 success-path runs: runtime p50 4.2 s, p95 4.4 s; 3/3 passed the quality bar.

## Failure paths (terminal-state correctness)

| Scenario | Expected | Got | Correct | Time |
| --- | --- | --- | --- | --- |
| retry loop exhausted | `FAILED_EVAL_EXHAUSTED` | `FAILED_EVAL_EXHAUSTED` | yes | 22.24 s |
| invalid input: 3-row CSV | `FAILED_INPUT` | `FAILED_INPUT` | yes | 7.62 s |
| guardrail: prompt injection | `FAILED_INPUT` | `FAILED_INPUT` | yes | 7.61 s |
| guardrail: missing target column | `FAILED_INPUT` | `FAILED_INPUT` | yes | 6.95 s |

Terminal state correct in 7/7 runs overall.

## Monthly cost scenario (LLM tokens)

Measured mean per successful run: 0 input + 0 output tokens. Cloud price used: $0.15/1M input, $1.25/1M output (Gemini Flash-Lite class). Deterministic fallbacks and cache hits cost $0.

| Runs per month | As measured (current mix of cloud / cache) | If every call went to the cloud model |
| --- | --- | --- |
| 100 | $0.00 | $0.00 |
| 1,000 | $0.00 | $0.00 |
| 10,000 | $0.00 | $0.00 |

Serving cost is separate: the local deployment runs inside the dashboard API at no cost; a Cloud Run service (1 GiB, scale to zero) is billed per request-second.

Raw per-run data: `reports/benchmark.json`.
