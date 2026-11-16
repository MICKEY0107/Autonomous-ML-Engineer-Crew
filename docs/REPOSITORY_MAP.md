# Autonomous ML Engineer Crew — Clear Repository File & Folder Map

This document explains **every directory and file** in this project in clear, plain language.

---

## 📂 Directory Layout

```
Autonomous ML Engineer Crew/
├── agents/             🤖 The 4 core AI agent implementations
├── contracts/          📋 Pydantic v2 schemas and data validation models
├── docs/               📚 Project documentation, architecture guides & explainer UI
├── orchestrator/       🎼 Flow controller, A2A client, telemetry, usage ledger
├── reports/            📊 Benchmark outputs, test evidence, and ADK eval HTML reports
├── scripts/            🛠️ Batch launchers, benchmark runner, and helper scripts
├── serving/            🚢 Fast prediction API container for cloud deployment
├── tests/              🧪 Automated pytest test suite (114 passing tests)
│   └── adk_eval/       🎯 Google ADK native evaluation set (faculty demo format)
├── tools/              🔧 FastMCP external tools (profiler & MLflow experiment logger)
├── ui/                 💻 React + TypeScript + Vite web dashboard
├── data/               📊 Sample CSV datasets for demo (house prices, iris, churn)
├── artifacts/          💾 Runtime outputs (models, SQLite DBs, split CSVs - gitignored)
├── mlruns/             📈 MLflow runtime tracking store (gitignored)
└── root files          🔑 Configuration, main CLI entrypoint, and project metadata
```

---

## 🔍 Detailed Breakdown by Folder

### 1. `agents/` — The AI Workers
- [`agents/data_prep.py`](agents/data_prep.py): **CrewAI Data Preparation Agent**. Calls the `profile_dataset` MCP tool, executes the LLM's data preparation plan (`DataPrepPlan`), handles imputation, splits train/test, and writes CSV artifacts.
- [`agents/model_selection_adk.py`](agents/model_selection_adk.py): **Google ADK Model Selection Agent**. Runs as an independent A2A service via `to_a2a()`. Plans search parameters, trains 3 models (Logistic Regression, Random Forest, Gradient Boosting for classification; Linear Regression, Random Forest, Ridge for regression), logs to MLflow via MCP, and selects the winner inside a 1% quality band.
- [`agents/evaluation.py`](agents/evaluation.py): **CrewAI Model Evaluation Agent**. Downloads winner from MLflow, evaluates on held-out test data, checks the deterministic threshold (F1 ≥ 0.80 or RMSE ≤ 25% std), and generates retry feedback if failed.
- [`agents/deployment.py`](agents/deployment.py): **CrewAI Deployment & Observability Agent**. Packages model bundle with SHA-256 digest, configures serving policy, and supports local, Docker, and Google Cloud Run deployment targets.

### 2. `contracts/` — The Structured Boundaries (Pydantic v2)
- [`contracts/models.py`](contracts/models.py): Defines the 6 core immutable contracts with `extra="forbid"`:
  1. `PipelineState` — Mutable state shared across stages.
  2. `DataPrepResult` — Data prep stage outcome.
  3. `ModelSelectionResult` — ADK agent response over A2A.
  4. `EvaluationResult` — Held-out scores and pass/retry feedback.
  5. `DeploymentResult` — Deployed endpoint, revision, and digest.
  6. `PipelineResult` — Terminal output JSON and Markdown model card.

### 3. `tools/` — Model Context Protocol (FastMCP)
- [`tools/profile_dataset.py`](tools/profile_dataset.py): **MCP Tool 1**. Reads CSV, calculates SHA-256, profiles nulls and data types, and flags data leakage risks without exposing raw rows.
- [`tools/log_experiment.py`](tools/log_experiment.py): **MCP Tool 2**. FastMCP interface to MLflow supporting `create_run`, `log_params`, `log_metrics`, `query_best_run`, and `register_model`.

### 4. `orchestrator/` — The Brain & Coordination Plane
- [`orchestrator/flow.py`](orchestrator/flow.py): **CrewAI Flow**. Routes stages using `@start`, `@listen`, and `@router`. Manages the retry loop (max 2 retries, 3 attempts) and fires 5 lifecycle callbacks.
- [`orchestrator/a2a_client.py`](orchestrator/a2a_client.py): Discovers ADK Agent Card at `/.well-known/agent-card.json` and sends `SendMessage` JSON-RPC over HTTP.
- [`orchestrator/telemetry.py`](orchestrator/telemetry.py): Langfuse and OpenTelemetry integration. Creates correlated spans for each agent and tool.
- [`orchestrator/usage.py`](orchestrator/usage.py): Tracks token consumption, latency, and estimated cost per decision.
- [`orchestrator/agentic_policy.py`](orchestrator/agentic_policy.py): Typed LLM decision schemas (`DataPrepPlan`, `ModelSearchPlan`, `ModelChoice`).
- [`orchestrator/crew_agents.py`](orchestrator/crew_agents.py): CrewAI Agent/Task/Crew configurations.

### 5. `tests/` — Automated Quality Assurance (114 Passing Tests)
- `tests/test_contracts.py`: 14 tests for Pydantic v2 schemas and validation rules.
- `tests/test_guardrails.py`: 16 tests for file limits, prompt injection, and toxic phrases.
- `tests/test_flow.py`: 5 tests for Flow routing, stage progression, and bounded retries.
- `tests/test_day2_integration.py`: End-to-end integration across all contracts.
- `tests/test_deployment.py`: 11 tests for model packaging and prediction serving.
- `tests/test_tracing.py`: OpenTelemetry trace hierarchy and W3C traceparent tests.
- `tests/test_adk_eval.py`: Pytest integration running the 6 native ADK eval cases.
- **`tests/adk_eval/`** (Faculty Demo Specification):
  - `evalset.json`: 6 evaluation cases matching `DEMO/ADK/Eval`.
  - `eval_config.json`: Trajectory and response matching thresholds.
  - `run_eval.py`: Standalone evaluation runner script.
  - `README.md`: Evaluation framework documentation.

### 6. `docs/` — User & Evaluator Guides
- [`docs/project-explainer.html`](docs/project-explainer.html): **Interactive Standalone HTML Guide**. Features dark-mode UI, architecture flow, agent cards, requirements audit, and concepts glossary.
- [`docs/minimum-requirements-compliance-audit.md`](docs/minimum-requirements-compliance-audit.md): Complete mapping of all 11 requirements to code.
- [`docs/faculty-demo-concept-comparison.md`](docs/faculty-demo-concept-comparison.md): Comparison with professor's 17 CrewAI notebooks and ADK demos.
- [`docs/architecture-alignment-verification.md`](docs/architecture-alignment-verification.md): Verification report confirming diagram alignment.

### 7. `reports/` — Evidence & Reports
- [`reports/evaluation_evidence.md`](reports/evaluation_evidence.md): Complete evidence report covering all 114 tests, benchmark, and guardrails.
- [`reports/eval_results_report.html`](reports/eval_results_report.html): **Visual HTML report** for ADK agent evaluation (matches `DEMO/ADK/Eval`).
- [`reports/benchmark.md`](reports/benchmark.md): Performance, runtime, cost, and terminal-state benchmark results.

### 8. `serving/` — Cloud Deployment Container
- `serving/app.py`: Standalone FastAPI service serving `/health`, `/schema`, and `/predict`.
- `serving/Dockerfile`: Container image definition used for Docker and Google Cloud Run.
- `serving/requirements.txt`: Minimal dependencies for model inference.

### 9. Key Root Files
- [`main.py`](main.py): Guarded command-line entrypoint (`--csv`, `--target`, `--goal`).
- [`orchestrator_api.py`](orchestrator_api.py): FastAPI backend that hosts the React dashboard and API endpoints.
- [`architecture_diagram_publication.png`](architecture_diagram_publication.png): Verified publication-ready system architecture diagram.
- `pyproject.toml` & `uv.lock`: Locked dependencies declaring CrewAI and Google ADK.
