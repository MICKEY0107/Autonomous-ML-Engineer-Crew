param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProjectId,

    [ValidateNotNullOrEmpty()]
    [string]$Region = "asia-south1",

    [ValidateNotNullOrEmpty()]
    [string]$ServiceName = "autonomous-ml-engineer-crew",

    [ValidateNotNullOrEmpty()]
    [string]$EvidenceBucket = "",

    [AllowEmptyString()]
    [string]$CloudRunVcpuUsdPerSecond = "",

    [AllowEmptyString()]
    [string]$CloudRunMemoryGibUsdPerSecond = ""
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
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com storage.googleapis.com logging.googleapis.com monitoring.googleapis.com | Out-Host

if (-not $EvidenceBucket) {
    $EvidenceBucket = "$ProjectId-amec-evaluation-evidence"
}

gcloud storage buckets describe "gs://$EvidenceBucket" --project $ProjectId 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gcloud storage buckets create "gs://$EvidenceBucket" --project $ProjectId --location $Region --uniform-bucket-level-access --public-access-prevention | Out-Host
}

foreach ($secret in $requiredSecrets) {
    gcloud secrets describe $secret --project $ProjectId 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Required Secret Manager secret '$secret' does not exist in project '$ProjectId'."
    }
}

$envVarList = @(
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
    "MLFLOW_UI_URL=/api/runs",
    "EVALUATION_EVIDENCE_BUCKET=$EvidenceBucket",
    "CLOUD_RUN_REGION=$Region",
    "GIT_COMMIT=$(git rev-parse --short HEAD)",
    "CLOUD_RUN_VCPU=2",
    "CLOUD_RUN_MEMORY_GIB=4",
    "LLM_INPUT_COST_PER_1M=0.15",
    "LLM_OUTPUT_COST_PER_1M=1.25"
)
if ($CloudRunVcpuUsdPerSecond) {
    $envVarList += "CLOUD_RUN_VCPU_USD_PER_SECOND=$CloudRunVcpuUsdPerSecond"
}
if ($CloudRunMemoryGibUsdPerSecond) {
    $envVarList += "CLOUD_RUN_MEMORY_GIB_USD_PER_SECOND=$CloudRunMemoryGibUsdPerSecond"
}
$envVars = $envVarList -join ","

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

$serviceAccount = gcloud run services describe $ServiceName `
    --project $ProjectId `
    --region $Region `
    --format "value(spec.template.spec.serviceAccountName)"
if (-not $serviceAccount) {
    $projectNumber = gcloud projects describe $ProjectId --format "value(projectNumber)"
    $serviceAccount = "$projectNumber-compute@developer.gserviceaccount.com"
}

gcloud storage buckets add-iam-policy-binding "gs://$EvidenceBucket" `
    --member "serviceAccount:$serviceAccount" `
    --role "roles/storage.objectCreator" `
    --quiet | Out-Host
gcloud storage buckets add-iam-policy-binding "gs://$EvidenceBucket" `
    --member "serviceAccount:$serviceAccount" `
    --role "roles/storage.objectViewer" `
    --quiet | Out-Host

gcloud run services update $ServiceName `
    --project $ProjectId `
    --region $Region `
    --update-env-vars "PUBLIC_BASE_URL=$serviceUrl" `
    --quiet | Out-Host

$evaluationJob = "$ServiceName-evaluation"
gcloud run jobs deploy $evaluationJob `
    --source . `
    --project $ProjectId `
    --region $Region `
    --service-account $serviceAccount `
    --command python `
    --args scripts/run_cloud_evaluation.py `
    --task-timeout 3600s `
    --max-retries 0 `
    --set-env-vars "EVALUATION_TARGET_URL=$serviceUrl,EVALUATION_EVIDENCE_BUCKET=$EvidenceBucket,CLOUD_RUN_REGION=$Region" `
    --quiet | Out-Host

$health = Invoke-RestMethod -Uri "$serviceUrl/api/health" -TimeoutSec 60
Write-Host "Cloud Run service: $serviceUrl"
Write-Host "Health: $($health | ConvertTo-Json -Compress)"
Write-Host "Evaluation bucket: gs://$EvidenceBucket"
Write-Host "Evaluation job: $evaluationJob"
Write-Host "Run it with: gcloud run jobs execute $evaluationJob --project $ProjectId --region $Region --wait"
