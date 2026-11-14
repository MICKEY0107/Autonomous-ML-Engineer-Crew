# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS ui-build
WORKDIR /src/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.12.8 AS uv

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    A2A_BIND_HOST=127.0.0.1 \
    A2A_BIND_PORT=8081 \
    A2A_AGENT_URL=http://127.0.0.1:8081 \
    A2A_TASKS_URL=http://127.0.0.1:8081/tasks \
    API_CORS_ORIGINS="" \
    MLFLOW_UI_URL=/api/runs \
    LANGFUSE_BASE_URL=https://cloud.langfuse.com \
    LANGFUSE_PROJECT_ID=unconfigured \
    LLM_PROVIDER=gemini \
    LLM_MODE=offline \
    CREWAI_MODEL=gemini/gemini-3.5-flash-lite \
    DEPLOY_TARGET=cloudrun_in_process \
    CREWAI_TRACING_ENABLED=false \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
COPY --from=ui-build /src/ui/dist ./ui/dist
RUN uv sync --frozen --no-dev \
    && mkdir -p /app/artifacts/ui /tmp/artifacts \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/artifacts /tmp/artifacts

USER appuser
EXPOSE 8080

CMD ["sh", "-c", "uvicorn orchestrator_api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
