# Day 4 Verification Report

**Verification date:** 2026-10-23  
**Workspace:** `C:\code\Autonomous ML Engineer Crew`

## Result summary

- Full pytest suite: **54 passed, 12 warnings in 33.86 seconds**.
- Normal CLI run: **SUCCESS** on the 500-row `data/house_prices.csv` dataset.
- Forced evaluation run: **FAILED_EVAL_EXHAUSTED after 2 retries**, as expected.
- Live Langfuse API verification: successful trace contains the pipeline root, all three agent spans, the profile MCP child span, and the A2A child span with the expected parent relationships.
- Forced-run trace contains three model-selection, three A2A, and three evaluation attempts. Persisted model-selection retry counts are `0`, `1`, and `2`.

## Verification checklist

✓ **1. All 6 Pydantic contracts instantiate and validate** — test name: `tests/test_day2_integration.py::test_all_contracts_instantiate`

✓ **2. `profile_dataset` returns a typed result with SHA-256** — test name: `tests/test_day2_integration.py::test_profile_dataset_tool`

✓ **3. `log_and_query_experiment` logs to MLflow and returns the best run** — test name: `tests/test_day2_integration.py::test_log_experiment_tool`

✓ **4. Data-prep agent produces train/test files on disk** — test name: `tests/test_day2_integration.py::test_data_prep_agent`

✓ **5. ADK agent responds to an A2A POST with a valid `ModelSelectionResult`** — test name: `tests/test_day2_integration.py::test_adk_agent_a2a`

✓ **6. Evaluation agent returns pass/fail with a metric and feedback** — test name: `tests/test_day2_integration.py::test_evaluation_agent`

✓ **7. Full pipeline runs on real file-backed data without crashing** — evidence: run ID `ef072b48-4c59-476e-9c4b-8fc896db01fe`; terminal state `SUCCESS`

✓ **8. Retry loop triggers on failure and `retry_count` increments** — evidence: [forced-run Langfuse trace](https://cloud.langfuse.com/trace/ce4103d9f11d4d0ba4dcd7d23aee4e30); live observation metadata contains model-selection retry counts `0 → 1 → 2`

✓ **9. `FAILED_EVAL_EXHAUSTED` occurs after the maximum retries** — evidence: terminal output `[PIPELINE]    ✗ FAILED_EVAL_EXHAUSTED after 2 retries`

✓ **10. All 9 callback invocations are correlated by `run_id`** — test name: `tests/test_flow.py::test_all_nine_callback_invocations_are_correlated_by_run_id`. This test records six successful-path events plus three validation-failure events and asserts that all nine carry the same `RUN_ID`. Retry callback behavior is independently verified by `tests/test_flow.py::test_flow_sends_feedback_and_succeeds_on_retry`.

✓ **11. Langfuse trace shows agent spans, MCP child span, and A2A span** — trace URL: [successful Langfuse trace](https://cloud.langfuse.com/trace/ef072b484c59476e9c4b8fc896db01fe). The live API confirmed:

- `agent.data_prep`, `agent.model_selection`, and `agent.evaluation` are children of `pipeline.run`.
- `mcp.profile_dataset` is a child of `agent.data_prep`.
- `a2a.tasks.send` is a child of `agent.model_selection`.
- Every non-root observation has populated input and output.

✓ **12. Guardrails block bad CSVs, prompt injection, and malformed MCP returns** — test names:

- `tests/test_guardrails.py::test_csv_guard_rejects_ten_or_fewer_rows`
- `tests/test_guardrails.py::test_csv_guard_rejects_path_traversal_and_100_mib_file`
- `tests/test_guardrails.py::test_prediction_goal_rejects_prompt_injection`
- `tests/test_guardrails.py::test_profile_mcp_result_is_strictly_validated`
- `tests/test_guardrails.py::test_experiment_mcp_result_is_strictly_validated`
- `tests/test_guardrails.py::test_llm_context_allows_metadata_and_rejects_raw_rows`

## Commands executed

### Full test suite

Langfuse network export was disabled only inside the pytest process so synthetic test spans would not pollute the live project. Application tracing remained enabled for both CLI verification runs.

```powershell
.\.venv\Scripts\python.exe -c "import os, pytest; os.environ['LANGFUSE_TRACING_ENABLED']='false'; raise SystemExit(pytest.main(['-v', '--tb=short']))"
```

Result:

```text
collected 54 items
tests\test_contracts.py ..............
tests\test_data_prep.py ...
tests\test_day2_integration.py ......
tests\test_evaluation.py ...
tests\test_flow.py .....
tests\test_guardrails.py .........
tests\test_log_experiment.py ...
tests\test_model_selection_adk.py ...
tests\test_profile_dataset.py ...
tests\test_tracing.py .....
54 passed, 12 warnings in 33.86s
```

The warnings are dependency notices: Google ADK A2A experimental-status warnings, CrewAI deprecations, one Pydantic Settings forward-reference warning, and one SQLAlchemy/MLflow deprecation warning. None caused a failure.

### Normal pipeline

```powershell
.\.venv\Scripts\python.exe main.py --csv data\house_prices.csv --target sale_price --goal "Predict residential sale price from property attributes."
```

Actual output:

```text
[DATA-PREP]   ✓ Profiled 500 rows, 9 features, 0 leakage risks
[MODEL-SEL]   ✓ Best model: Ridge (rmse=19,919.00) after 1 attempt
[EVAL]        ✓ PASSED — rmse=18,271.44 is below threshold 53,204.02
[PIPELINE]    ✓ SUCCESS — run_id: ef072b48-4c59-476e-9c4b-8fc896db01fe
[TRACE]       https://cloud.langfuse.com/trace/ef072b484c59476e9c4b8fc896db01fe
```

### Forced evaluation failure

```powershell
.\.venv\Scripts\python.exe main.py --csv data\house_prices.csv --target sale_price --goal "Predict residential sale price from property attributes." --force-fail-eval
```

Actual output:

```text
[DATA-PREP]   ✓ Profiled 500 rows, 9 features, 0 leakage risks
[MODEL-SEL]   ✓ Best model: Ridge (rmse=19,919.00) after 1 attempt
[EVAL]        ✗ FAILED — rmse=18,271.44 exceeds threshold 0.99 | retry 1/2
[MODEL-SEL]   ✓ Best model: Ridge (rmse=19,951.78) after 2 attempts
[EVAL]        ✗ FAILED — rmse=18,350.00 exceeds threshold 0.99 | retry 2/2
[MODEL-SEL]   ✓ Best model: LinearRegression (rmse=19,959.34) after 3 attempts
[PIPELINE]    ✗ FAILED_EVAL_EXHAUSTED after 2 retries
[TRACE]       https://cloud.langfuse.com/trace/ce4103d9f11d4d0ba4dcd7d23aee4e30
```

## Trace API evidence

The Langfuse v2 observations API returned these counts for the successful run:

```text
pipeline.run: 1
agent.data_prep: 1
mcp.profile_dataset: 1
agent.model_selection: 1
a2a.tasks.send: 1
agent.evaluation: 1
```

Exact parent checks all returned `True`:

```text
agent.data_prep parent is pipeline.run
mcp.profile_dataset parent is agent.data_prep
agent.model_selection parent is pipeline.run
a2a.tasks.send parent is agent.model_selection
agent.evaluation parent is pipeline.run
```

The forced-run trace returned three observations each for `agent.model_selection`, `a2a.tasks.send`, and `agent.evaluation`, with persisted model-selection retry metadata values `0`, `1`, and `2`.
