# Autonomous ML Engineer Crew

A typed, observable multi-agent pipeline for tabular machine learning. The
project accepts a CSV and prediction goal, prepares the data, compares three
candidate models through a Google ADK A2A agent, evaluates the winner on held-out
data, and either succeeds or retries with feedback.

The system combines a CrewAI orchestration flow, Google ADK, MCP tools, MLflow,
Langfuse/OpenTelemetry, Pydantic v2 contracts, a FastAPI boundary, and a React
dashboard.

![Autonomous ML Engineer Crew architecture](architecture_diagram_publication.png)

## What it does

```text
CSV + target + prediction goal
        │
        ▼
Input guardrails
        │
        ▼
Data-prep agent ──► profile_dataset MCP tool
        │
        ▼
ADK model-selection agent over A2A ──► MLflow MCP tool
        │
        ▼
Held-out evaluation
        │
        ├── pass ─────────────────────────► SUCCESS
        │
        └── fail ─► retry with feedback ──► maximum two retries
                                               │
                                               └── FAILED_EVAL_EXHAUSTED
```

Key capabilities:

- Supports regression and classification datasets.
- Uses Pydantic v2 models at every agent and tool boundary.
- Trains exactly three task-appropriate candidate models.
- Executes typed LLM data-preparation plans after deterministic validation.
- Lets the LLM configure bounded hyperparameters for all three candidates and
  apply retry feedback to the next search attempt.
- Lets the LLM choose among trained candidates inside a deterministic quality
  band, so reasoning affects the selected artifact instead of merely narrating it.
- Persists experiment parameters, metrics, artifacts, and winners in MLflow.
- Sends the model-selection task through a local Google ADK A2A service.
- Produces correlated Langfuse/OpenTelemetry spans without logging raw rows.
- Retries failed evaluations at most twice, for three total attempts.
- Exposes live progress, contracts, results, and trace structure in a React UI.
- Fails closed on invalid CSVs, prompt-injection-like goals, and malformed tool
  responses.

## Agentic decisions versus deterministic execution

The LLM is the policy layer. It receives row-free schemas, quality metadata,
candidate metrics, timings, the prediction goal, and retry feedback. It returns
Pydantic-validated plans that the pipeline executes:

- data-prep exclusions, split size, and imputation strategy;
- candidate-model hyperparameters, revised on retries using evaluation feedback;
- model choice among candidates that meet the quality band;
- evaluation interpretation, improvement advice, and the final run summary.

Deterministic code remains responsible for guardrails, CSV access, leakage
enforcement, splitting, transformations, model fitting, metric computation,
artifact writes, contract validation, and the final pass/fail threshold. Invalid
or unavailable LLM output falls back to a safe typed policy. Set `LLM_MODE=live`
for real decisions or `LLM_MODE=offline` for repeatable local tests.

## Technology stack

| Area | Technology |
|---|---|
| Orchestration | CrewAI Flow |
| Remote agent | Google ADK exposed through A2A on `127.0.0.1:8080` |
| Tool protocol | MCP |
| Contracts | Pydantic v2 |
| Model training | pandas and scikit-learn |
| Experiment tracking | MLflow |
| Tracing | Langfuse and OpenTelemetry |
| Backend | FastAPI and Uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Tests | pytest |

## Requirements

- Windows 10/11, macOS, or Linux
- Python `3.11` through `3.13`
- Node.js `20+` and npm
- Git is optional for local execution

The commands below use PowerShell because the project is currently developed
and verified on Windows.

## Quick start

### 1. Create the Python environment

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

If the existing `.venv` is already populated, this step can be skipped.

### 2. Install and build the dashboard

```powershell
cd ui
npm install
npm run build
cd ..
```

The build is required because FastAPI serves `ui/dist` at the application root.

### 3. Start the application

```powershell
.\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

The same process serves both the dashboard and `/api` endpoints. The ADK A2A
server is started and stopped automatically for each pipeline execution.

### 4. Run a pipeline

1. Select **Drop a CSV here or browse**.
2. Choose the target column.
3. Enter a prediction goal.
4. Optionally enable **Retry Loop Demo**.
5. Select **Run pipeline**.
6. Watch the nine-stage timeline and inspect the Pipeline, Contracts, and Traces
   tabs when the run completes.

The included demo dataset is `data/house_prices.csv`, with `sale_price` as its
target column.

## Configuration

The telemetry module automatically loads `.env` from the project root.

```dotenv
LLM_PROVIDER=gemini
LLM_MODE=live
LLM_MAX_TOKENS=700
LLM_FALLBACK_TIMEOUT_SECONDS=15
GEMINI_API_KEY=your-google-ai-studio-key
CREWAI_MODEL=gemini/gemini-3.8-flash

LANGFUSE_PUBLIC_KEY=pk-lf-your-key
LANGFUSE_SECRET_KEY=sk-lf-your-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PROJECT_ID=your-project-id

A2A_TASKS_URL=http://localhost:8080/a2a/tasks/send
A2A_BIND_HOST=127.0.0.1
A2A_BIND_PORT=8080
MLFLOW_UI_URL=http://127.0.0.1:5001
API_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
VITE_API_PROXY_TARGET=http://127.0.0.1:8000
```

Langfuse and LLM credentials are optional. Without Langfuse credentials, local
OpenTelemetry data remains available. Without the selected LLM provider key,
the deterministic ML pipeline still runs and its reasoning panels use local,
row-safe fallback explanations. Copy `.env.example` to `.env`, then replace
only the credential placeholders needed for your environment.

For OpenRouter, `OPENROUTER_API_KEY` is preferred; the compatibility alias
`OPEN_ROUTER_API_KEY` is also supported.

When both Gemini and OpenRouter credentials are present, a transport or quota
failure on the selected provider is retried once through the other provider
before the typed deterministic fallback is used. Optional
`GEMINI_FALLBACK_MODEL` and `OPENROUTER_FALLBACK_MODEL` values override the
model name derived from `CREWAI_MODEL`. `LLM_MAX_TOKENS` bounds decision output
cost, while `LLM_FALLBACK_TIMEOUT_SECONDS` prevents a secondary provider from
stalling the pipeline.

For standalone MLflow tool calls, `MLFLOW_TRACKING_URI` can also be set:

```dotenv
MLFLOW_TRACKING_URI=sqlite:///artifacts/e2e/mlflow.db
```

Never commit real secret keys.

## Running MLflow

Dashboard executions share `artifacts/ui/mlflow.db`. Start its UI with:

```powershell
.\scripts\start_dashboard_mlflow.bat
```

Then open <http://127.0.0.1:5001>.

The end-to-end scripts use `artifacts/e2e/mlflow.db`. To inspect that store, run:

```powershell
.\scripts\start_mlflow.bat
```

Only one process can use port `5001` at a time.

## Command-line usage

The pipeline can run without the dashboard:

```powershell
.\.venv\Scripts\python.exe main.py `
  --csv data/house_prices.csv `
  --target sale_price `
  --goal "predict house price"
```

To demonstrate the bounded retry path:

```powershell
.\.venv\Scripts\python.exe main.py `
  --csv data/house_prices.csv `
  --target sale_price `
  --goal "predict house price" `
  --force-fail-eval
```

`--force-fail-eval` sets the evaluation threshold to `0.99`. It is intended for
demonstrating retry behavior, not normal model acceptance.

## End-to-end demo scripts

Successful run:

```powershell
.\.venv\Scripts\python.exe scripts\run_e2e.py
```

Forced evaluation failure with two retries:

```powershell
.\.venv\Scripts\python.exe scripts\run_forced_fail.py
```

The setup attempts to load a real Titanic dataset and falls back to generating
a realistic 500-row house-price dataset when network access is unavailable.

## Development mode

For frontend work, run the backend and Vite development server separately.

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --reload --port 8000
```

Terminal 2:

```powershell
cd ui
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` requests to port `8000`.

After frontend changes, rebuild `ui/dist` before returning to the single-server
URL on port `8000`:

```powershell
cd ui
npm run build
```

## Pipeline stages

### 1. Input validation

The input boundary checks the CSV, target column, and prediction goal. Invalid
input returns a typed `FAILED_INPUT` result instead of crashing the process.

### 2. Data preparation

`agents/data_prep.py` validates and reads the CSV, drops rows with missing target
values, removes duplicates, prepares the feature set, and creates an 80/20
train/test split. Classification splits are stratified when the dataset permits
it. The resulting CSV artifacts are written under the active artifact root.

### 3. Dataset profiling MCP tool

`tools/profile_dataset.py` returns a validated profile containing:

- column schema;
- null counts;
- leakage candidates;
- row and feature metadata; and
- a SHA-256 digest of the source CSV.

### 4. ADK model selection over A2A

`agents/model_selection_adk.py` receives the typed data-prep result through an
A2A task message and trains three candidates:

| Task | Candidates |
|---|---|
| Classification | Logistic Regression, Random Forest, Gradient Boosting |
| Regression | Linear Regression, Random Forest, Ridge |

Candidate runs are logged through the MLflow MCP tool. The best run is selected,
its model is registered, and a `ModelSelectionResult` is returned through A2A.

### 5. Held-out evaluation

`agents/evaluation.py` loads the selected model and evaluates it on the test
artifact that was not used for final training.

- Classification uses weighted F1 and passes by default at `0.80` or higher.
- Regression uses RMSE and passes when RMSE is no greater than 25% of the held-out
  target standard deviation, unless an explicit threshold is provided.

The result contains the metric values, pass/fail decision, threshold rationale,
and actionable feedback.

### 6. Retry or terminal result

When evaluation fails and fewer than two retries have occurred, the flow sends
evaluation feedback back to the ADK agent and repeats model selection and
evaluation. Terminal states are:

| State | Meaning |
|---|---|
| `SUCCESS` | The held-out evaluation passed. |
| `FAILED_INPUT` | Input or guardrail validation failed. |
| `FAILED_EVAL_EXHAUSTED` | Evaluation still failed after two retries. |

## Contracts

All orchestration boundaries use models from `contracts/models.py`:

- `PipelineState`
- `DataPrepResult`
- `ModelSelectionResult`
- `EvaluationResult`
- `DeploymentResult`
- `PipelineResult`

Tool-specific results, including `ProfileResult` and `ExperimentResult`, are also
Pydantic models. External data is reparsed at each boundary so malformed tool or
A2A responses fail closed.

## Guardrails and data handling

The project enforces these controls before and during execution:

- CSV size must be less than 100 MiB.
- A dataset must contain more than 10 data rows.
- Path traversal is rejected.
- The requested target column must exist.
- Prediction goals are checked for prompt-injection-like instructions.
- Raw rows are excluded from LLM prompts and telemetry.
- MCP and A2A return values are validated with Pydantic before use.
- Langfuse span payloads contain schema, metrics, hashes, and artifact URIs—not
  raw row arrays.

## Observability

Each pipeline uses its `run_id` to correlate callbacks, artifacts, MLflow runs,
and traces. The trace hierarchy is:

```text
pipeline.run
├── agent.data_prep
│   └── mcp.profile_dataset
├── agent.model_selection
│   ├── a2a.tasks.send
│   └── mcp.log_and_query_experiment
└── agent.evaluation
```

Span inputs and outputs are populated with redacted, row-safe metadata. The
dashboard presents the same hierarchy in its Traces tab and links to the
project-scoped Langfuse trace when credentials are configured.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serve the compiled React dashboard. |
| `GET` | `/api/health` | Liveness check. |
| `POST` | `/api/pipeline/run` | Upload a CSV and start a background run. |
| `GET` | `/api/pipeline/status` | Read live stage and retry state. |
| `GET` | `/api/pipeline/result` | Read the completed result and trace tree. |
| `GET` | `/docs` | FastAPI interactive API documentation. |

The run endpoint expects multipart fields named `csv_file`, `target_column`,
`prediction_goal`, and `force_fail_eval`.

## Testing

Run the full Python suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v --tb=short
```

Run the frontend checks:

```powershell
cd ui
npm run typecheck
npm run build
npm audit --omit=dev
```

Tests cover contracts, both MCP tools, data preparation, A2A model selection,
evaluation, retries, callbacks, guardrails, tracing, and the dashboard API
contract.

## Project structure

```text
.
├── contracts/
│   └── models.py                  # Shared Pydantic v2 contracts
├── tools/
│   ├── profile_dataset.py         # Dataset-profile MCP tool
│   └── log_experiment.py          # MLflow MCP tool
├── agents/
│   ├── data_prep.py               # CrewAI data-prep agent
│   ├── model_selection_adk.py     # Google ADK/A2A model-selection agent
│   └── evaluation.py              # CrewAI held-out evaluator
├── orchestrator/
│   ├── flow.py                    # CrewAI flow and retry controller
│   └── telemetry.py               # Langfuse/OpenTelemetry integration
├── ui/                            # React dashboard
├── tests/                         # Unit and integration tests
├── scripts/
│   ├── run_e2e.py                 # Successful end-to-end run
│   ├── run_forced_fail.py         # Retry-exhaustion demonstration
│   ├── start_dashboard_mlflow.bat # Dashboard MLflow store
│   └── start_mlflow.bat           # E2E MLflow store
├── data/
│   └── house_prices.csv           # Included demonstration dataset
├── orchestrator_api.py            # FastAPI dashboard facade
├── main.py                        # CLI entry point
└── pyproject.toml                 # Python dependencies and pytest config
```

## Generated files

Normal runs create local data beneath these directories:

- `artifacts/ui/` — dashboard uploads, train/test files, models, and MLflow DB;
- `artifacts/e2e/` — end-to-end script artifacts and MLflow DB;
- `mlruns/` — default MLflow artifacts for standalone tool calls; and
- `/tmp/artifacts/` or the platform temporary directory — agent test artifacts.

These directories can grow as more experiments are executed.

## Troubleshooting

### Port 8000 is already in use

Stop the existing Uvicorn terminal with `Ctrl+C`, then start the application
again. On Windows, the listener can be identified with:

```powershell
netstat -ano | Select-String ':8000'
```

### The root URL returns `{"detail":"Not Found"}`

An older API process is still running. Stop it, rebuild the dashboard, and start
Uvicorn again:

```powershell
cd ui
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --port 8000
```

### The CSV picker appears unchanged

Use `Ctrl+Shift+R` to refresh the compiled frontend. The Run button stays disabled
until a CSV, target column, and prediction goal are all present.

### The A2A port is busy

Port `8080` is used by the local ADK service. Stop the older Python/A2A process
before starting another pipeline.

### Langfuse is unavailable

Verify the three `LANGFUSE_*` values in `.env`. The pipeline itself continues to
work without cloud tracing.

### MLflow UI does not open

Start the correct store script and confirm that port `5001` is free. The dashboard
and E2E scripts use different SQLite files.

## Verification reports

- [`reports/frontend_backend_verification.md`](reports/frontend_backend_verification.md)
- [`reports/demo_verification.md`](reports/demo_verification.md)
- [`reports/day4_verification.md`](reports/day4_verification.md)

