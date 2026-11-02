# End-to-end benchmark

Generated 2026-11-02 04:23 UTC · 3 runs per dataset · LLM_MODE=live · LLM_CACHE=off · primary model `gemini/gemini-3.5-flash-lite` · fallback chain `ollama/qwen2.5-coder:latest,openrouter/google/gemini-3.5-flash-lite`

Each run is a full `main.py` execution: guardrails, CrewAI Data-Prep agent, MCP profiling, A2A call to the Google ADK model-selection agent, MLflow logging through MCP, CrewAI evaluation, bounded retry loop and the Deployment agent.

## Success paths

| Scenario | Runs | Passed | Runtime p50 | Runtime p95 | Mean LLM calls | Local / cached / rule-based | Mean tokens | Mean cost / run | Mean held-out metric | Deployed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| house prices (regression) | 3 | 3/3 | 21.9 s | 23.1 s | 6.0 | 1.0 / 0.0 / 0.0 | 4,106 | $0.00146 | rmse 18,271.4424 | 3/3 |
| customer churn (classification) | 3 | 3/3 | 22.7 s | 23.7 s | 6.0 | 1.0 / 0.0 / 0.0 | 4,232 | $0.00153 | f1_weighted 0.898 | 3/3 |
| iris (3-class) | 3 | 3/3 | 22.1 s | 24.4 s | 6.0 | 1.0 / 0.0 / 0.0 | 4,123 | $0.00146 | f1_weighted 0.9333 | 3/3 |

Across all 9 success-path runs: runtime p50 22.1 s, p95 24.4 s; 9/9 passed the quality bar.

## Failure paths (terminal-state correctness)

| Scenario | Expected | Got | Correct | Time |
| --- | --- | --- | --- | --- |
| retry loop exhausted | `FAILED_EVAL_EXHAUSTED` | `FAILED_EVAL_EXHAUSTED` | yes | 51.74 s |
| invalid input: 3-row CSV | `FAILED_INPUT` | `FAILED_INPUT` | yes | 8.16 s |
| guardrail: prompt injection | `FAILED_INPUT` | `FAILED_INPUT` | yes | 8.24 s |
| guardrail: missing target column | `FAILED_INPUT` | `FAILED_INPUT` | yes | 8.34 s |

Terminal state correct in 13/13 runs overall.

## Monthly cost scenario (LLM tokens)

Measured mean per successful run: 3,320 input + 833 output tokens. Cloud price used: $0.15/1M input, $1.25/1M output (Gemini Flash-Lite class). Local Ollama calls and cache hits cost $0.

| Runs per month | As measured (current mix of cloud / local / cache) | If every call went to the cloud model |
| --- | --- | --- |
| 100 | $0.15 | $0.15 |
| 1,000 | $1.49 | $1.54 |
| 10,000 | $14.86 | $15.40 |

Serving cost is separate: the local deployment runs inside the dashboard API at no cost; a Cloud Run service (1 GiB, scale to zero) is billed per request-second.

Raw per-run data: `reports/benchmark.json`.
