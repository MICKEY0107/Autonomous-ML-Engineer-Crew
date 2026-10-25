import axios from "axios";
import type {
  PipelineResultResponse,
  PipelineStatusResponse,
  RunInput,
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

export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    return typeof detail === "string" ? detail : error.message;
  }
  return error instanceof Error ? error.message : "Unexpected dashboard error";
}
