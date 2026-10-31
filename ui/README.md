# Autonomous ML Engineer Crew Dashboard

The React dashboard is a local UI wrapper around the existing typed CrewAI
pipeline. It never sends raw dataset rows to the UI or telemetry surfaces.

## Run locally

From the repository root, start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --reload --port 8000
```

In a second terminal, start the dashboard:

```powershell
cd ui
npm install
npm run dev
```

Open <http://localhost:5173>, upload a CSV, choose its target column, and run
the pipeline. The Vite development server proxies `/api` to port 8000.

## One-origin demo

Build the frontend once, then the FastAPI server also serves it at
<http://127.0.0.1:8000>. This is the address to use when you want a single
frontend/backend origin:

```powershell
cd ui
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --port 8000
```

To enable the dashboard's **View MLflow** link, run:

```powershell
.\scripts\start_dashboard_mlflow.bat
```

This serves the shared dashboard experiment store at <http://127.0.0.1:5055>.
The launcher uses one worker for reliable local startup on Windows.

## Production build

```powershell
cd ui
npm run build
```

The generated `dist/` bundle is static and can be served by any web server as
long as `/api` is proxied to the FastAPI process.
