# Frontend–Backend Integration Verification

**Date:** 2026-10-24  
**Overall status:** **READY FOR DEMO**

## Verified execution paths

| Check | Result | Evidence |
|---|---:|---|
| Frontend request matches backend API | ✓ | `test_dashboard_run_accepts_frontend_multipart_payload` posts the exact multipart fields used by the React form and receives HTTP 202. |
| Status polling contract | ✓ | `test_dashboard_status_contract_matches_frontend` verifies all nine ordered stages and the public status fields. |
| Same-origin dashboard route | ✓ | `GET /` now returned HTTP 200 with the React title; its compiled JavaScript asset, favicon, and `/api/health` also returned HTTP 200 from port 8000. |
| Live duration updates | ✓ | `test_running_stage_duration_advances_without_leaking_internal_clock` verifies running timers advance while the private clock field stays out of API responses. |
| Browser success path | ✓ | Uploaded `data/house_prices.csv`; the dashboard reached `SUCCESS` with run ID `1fab9bd4-099a-4d9d-ad30-971f5db7fc69`. |
| Browser reload recovery | ✓ | Reloaded the page after execution; the dashboard hydrated the final result, nine-stage timeline, trace, and metrics from `/api/pipeline/status` and `/api/pipeline/result`. |
| Browser retry/failure path | ✓ | Retry Loop Demo reached `FAILED_EVAL_EXHAUSTED`, displayed retry count `2/2`, and showed three model-selection and three evaluation attempts. Run ID: `0adafba4-1ccd-456c-ad89-ea08a74ec688`. |
| Contracts view | ✓ | All six contract cards render: `PipelineState`, `DataPrepResult`, `ModelSelectionResult`, `EvaluationResult`, `DeploymentResult`, and `PipelineResult`. |
| Trace view | ✓ | Root `pipeline.run` renders populated input/output; agent spans render beneath it; expanding data prep shows `mcp.profile_dataset`. No raw dataset rows were exposed. |
| Artifact handoff | ✓ | Success-run `train.csv` and `test.csv` both exist under `artifacts/ui/1fab9bd4099a4d9dad30971f5db7fc69/...`. |
| MLflow handoff | ✓ | Dashboard runs use `artifacts/ui/mlflow.db`; it contains 3 experiments and 12 runs. `scripts/start_dashboard_mlflow.bat` serves the UI on port 5001 and returned HTTP 200. |
| Browser runtime health | ✓ | Playwright console: 0 errors, 0 warnings. |
| CSV picker | ✓ | The visible upload zone opens the native chooser; selecting `house_prices.csv` populates the target-column selector and enables Run pipeline. |
| Python verification | ✓ | Full suite: **59 passed, 0 failed** in 25.20 seconds. |
| Frontend verification | ✓ | TypeScript typecheck and Vite production build completed successfully. |
| Production dependency audit | ✓ | `npm audit --omit=dev`: 0 vulnerabilities across all severities. |
| Guardrails | ✓ | Ten guardrail tests passed, covering row limits, path traversal, 100 MiB input, prompt injection, strict MCP validation, raw-row exclusion, and `FAILED_INPUT`. |

## Demo endpoints

- Dashboard: <http://127.0.0.1:5173>
- Backend health: <http://127.0.0.1:8000/api/health>
- MLflow UI: <http://127.0.0.1:5001> (run `scripts/start_dashboard_mlflow.bat` first)
- Success trace: <https://cloud.langfuse.com/project/cmvjt4kam04c9ad0jy0a92aum/traces/1fab9bd4099a4d9dad30971f5db7fc69>
- Retry trace: <https://cloud.langfuse.com/project/cmvjt4kam04c9ad0jy0a92aum/traces/0adafba41ccd456cad89ea08a74ec688>

## Integration fixes included in this pass

1. The React dashboard now hydrates an active or completed pipeline from the backend when the page loads, so refreshes do not lose execution state.
2. Running stage durations are calculated on every status snapshot and internal timing fields are never returned to the browser.
3. Dashboard executions now share one MLflow database, matching the **View MLflow** link.
4. The one-click dashboard MLflow launcher uses one worker, avoiding Windows multiprocess socket errors.
5. Three API integration tests now lock the frontend/backend boundary in place.
6. FastAPI now serves the compiled React dashboard at `/`, its favicon at `/favicon.svg`, and its assets at `/assets`, removing the prior root-path 404 and browser console error.
7. The CSV upload zone now uses a native label/file-input association rather than a simulated button; it supports mouse, keyboard, drag-and-drop, and reselection of the same file.

## Non-blocking upstream warnings

- Google ADK reports that its A2A adapter is experimental.
- CrewAI emits deprecation warnings for existing agent options.
- MLflow warns that its optional job-execution backend is unavailable on Windows; experiment tracking and the UI are working normally.

No frontend/backend integration blocker remains in the verified local demo path.
