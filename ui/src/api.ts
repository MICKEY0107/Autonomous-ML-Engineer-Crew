import axios from "axios";
import type {
  DeploymentMetadata,
  GuardrailCheck,
  PipelineResultResponse,
  PipelineStatusResponse,
  Prediction,
  ProvidersResponse,
  ResultPayload,
  RunInput,
  RunSummary,
  SampleDataset,
} from "./types";

const api = axios.create({
  baseURL: "/api",
  timeout: 15_000,
});

export async function startPipeline(input: RunInput): Promise<PipelineStatusResponse> {
  const form = new FormData();
  form.append("csv_file", input.file);
  form.append("target_column", input.targetColumn);
  form.append("prediction_goal", input.predictionGoal);
  form.append("force_fail_eval", String(input.forceFailEval));
  const response = await api.post<PipelineStatusResponse>("/pipeline/run", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120_000,
  });
  return response.data;
}

export async function getPipelineStatus(): Promise<PipelineStatusResponse> {
  const response = await api.get<PipelineStatusResponse>("/pipeline/status");
  return response.data;
}

export async function getPipelineResult(): Promise<PipelineResultResponse> {
  const response = await api.get<PipelineResultResponse>("/pipeline/result");
  return response.data;
}

export async function getProviders(refresh = false): Promise<ProvidersResponse> {
  const response = await api.get<ProvidersResponse>("/providers", { params: { refresh }, timeout: 30_000 });
  return response.data;
}

export async function checkGuardrail(goal: string): Promise<GuardrailCheck> {
  const form = new FormData();
  form.append("prediction_goal", goal);
  const response = await api.post<GuardrailCheck>("/guardrails/check", form);
  return response.data;
}

export async function listRuns(): Promise<RunSummary[]> {
  const response = await api.get<RunSummary[]>("/runs");
  return response.data;
}

export async function getRun(runId: string): Promise<ResultPayload> {
  const response = await api.get<ResultPayload>(`/runs/${runId}`);
  return response.data;
}

export function downloadUrl(runId: string, kind: "model-card.md" | "result.json" | "insights.json"): string {
  return `/api/runs/${runId}/download/${kind}`;
}

export async function getSamples(): Promise<SampleDataset[]> {
  const response = await api.get<SampleDataset[]>("/samples");
  return response.data;
}

export async function fetchSampleFile(name: string): Promise<File> {
  const response = await api.get<Blob>(`/samples/${name}`, { responseType: "blob" });
  return new File([response.data], name, { type: "text/csv" });
}

function keyHeader(apiKey?: string): Record<string, string> {
  return apiKey ? { "X-API-Key": apiKey } : {};
}

export async function getDeployment(runHex: string): Promise<DeploymentMetadata> {
  const response = await api.get<DeploymentMetadata>(`/deployments/${runHex}`);
  return response.data;
}

export async function predictRows(
  runHex: string,
  rows: Record<string, string | number | null>[],
  apiKey?: string,
): Promise<Prediction[]> {
  const response = await api.post<{ predictions: Prediction[] }>(
    `/deployments/${runHex}/predict`,
    { rows },
    { headers: keyHeader(apiKey) },
  );
  return response.data.predictions;
}

export async function predictCsv(runHex: string, file: File, apiKey?: string): Promise<Blob> {
  const form = new FormData();
  form.append("csv_file", file);
  const response = await api.post<Blob>(`/deployments/${runHex}/predict_csv`, form, {
    headers: { ...keyHeader(apiKey), "Content-Type": "multipart/form-data" },
    responseType: "blob",
    timeout: 120_000,
  });
  return response.data;
}

export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    return typeof detail === "string" ? detail : error.message;
  }
  return error instanceof Error ? error.message : "Unexpected dashboard error";
}
