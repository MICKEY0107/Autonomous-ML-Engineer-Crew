param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProjectId,

    [ValidateNotNullOrEmpty()]
    [string]$Region = "asia-south1",

    [ValidateNotNullOrEmpty()]
    [string]$ServiceName = "autonomous-ml-engineer-crew"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "Google Cloud CLI is not installed or is not on PATH."
}

$requiredSecrets = @(
    "amec-gemini-api-key",
    "amec-langfuse-public-key",
    "amec-langfuse-secret-key",
    "amec-langfuse-project-id"
)

gcloud config set project $ProjectId | Out-Host
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com | Out-Host

foreach ($secret in $requiredSecrets) {
    gcloud secrets describe $secret --project $ProjectId 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Required Secret Manager secret '$secret' does not exist in project '$ProjectId'."
    }
}

$envVars = @(
    "LLM_PROVIDER=gemini",
    "LLM_MODE=live",
    "CREWAI_MODEL=gemini/gemini-3.5-flash-lite",
    "LLM_FALLBACK_CHAIN=",
    "LLM_NARRATIVE_MODEL=",
    "LLM_CACHE=off",
    "LANGFUSE_BASE_URL=https://cloud.langfuse.com",
    "A2A_BIND_HOST=127.0.0.1",
    "A2A_BIND_PORT=8081",
    "A2A_AGENT_URL=http://127.0.0.1:8081",
    "A2A_TASKS_URL=http://127.0.0.1:8081/tasks",
    "A2A_TIMEOUT_SECONDS=360",
    "DEPLOY_TARGET=cloudrun_in_process",
    "API_CORS_ORIGINS=",
    "MLFLOW_UI_URL=/api/runs"
) -join ","

gcloud run deploy $ServiceName `
    --source . `
    --project $ProjectId `
    --region $Region `
    --allow-unauthenticated `
    --port 8080 `
    --cpu 2 `
    --memory 4Gi `
    --timeout 3600 `
    --concurrency 1 `
    --min-instances 1 `
    --max-instances 1 `
    --no-cpu-throttling `
    --set-env-vars $envVars `
    --set-secrets "GEMINI_API_KEY=amec-gemini-api-key:latest,LANGFUSE_PUBLIC_KEY=amec-langfuse-public-key:latest,LANGFUSE_SECRET_KEY=amec-langfuse-secret-key:latest,LANGFUSE_PROJECT_ID=amec-langfuse-project-id:latest" `
    --quiet | Out-Host

$serviceUrl = gcloud run services describe $ServiceName `
    --project $ProjectId `
    --region $Region `
    --format "value(status.url)"

if (-not $serviceUrl) {
    throw "Cloud Run did not return a service URL."
}

gcloud run services update $ServiceName `
    --project $ProjectId `
    --region $Region `
    --update-env-vars "PUBLIC_BASE_URL=$serviceUrl" `
    --quiet | Out-Host

$health = Invoke-RestMethod -Uri "$serviceUrl/api/health" -TimeoutSec 60
Write-Host "Cloud Run service: $serviceUrl"
Write-Host "Health: $($health | ConvertTo-Json -Compress)"
