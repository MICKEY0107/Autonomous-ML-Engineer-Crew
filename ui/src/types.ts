export type PipelineApiStatus = "idle" | "running" | "success" | "failed";
export type StageStatus = "pending" | "running" | "success" | "failed" | "retry";
export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type ReasoningData = Record<string, JsonValue>;

export interface PipelineStage {
  name: string;
  duration_ms: number;
  status: StageStatus;
  detail: string;
  attempts: number;
  reasoning?: ReasoningData;
}

export interface FinalResult {
  status: "SUCCESS" | "FAILED_EVAL_EXHAUSTED" | "FAILED_INPUT";
  run_id: string;
  model_name: string | null;
  metric_value: number | null;
  metric_name: string | null;
  threshold: number | null;
  feedback: string | null;
  token_count: number;
  estimated_cost_usd?: number;
  usage_report?: UsageReport;
  runtime_ms: number;
  trace_url: string;
  mlflow_run_id: string | null;
  mlflow_url: string;
  retry_count: number;
  error: string | null;
  completed_at: string;
  llm_summary: string;
}

export interface UsageAgentBucket {
  total_tokens: number;
  requests: number;
  estimated_cost_usd: number;
  latency_seconds: number;
}

export interface UsageReport {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  llm_requests?: number;
  llm_latency_seconds?: number;
  estimated_cost_usd?: number;
  pricing?: { input_per_1m_usd: number; output_per_1m_usd: number };
  by_agent?: Record<string, UsageAgentBucket>;
}

export interface LlmInsights {
  data_prep: string;
  model_selection: string;
  evaluation: string;
  summary: string;
}

export interface TraceNode {
  name: string;
  label?: string;
  status: string;
  duration_ms?: number;
  input?: unknown;
  output?: unknown;
  run_id?: string;
  children?: TraceNode[];
}

export interface PipelineStatusResponse {
  status: PipelineApiStatus;
  current_stage: string | null;
  retry_count: number;
  stages_completed: PipelineStage[];
  stages?: PipelineStage[];
  final_result: FinalResult | null;
  trace_data: TraceNode | null;
  error: string | null;
  started_at: string | null;
  updated_at: string;
  llm_insights?: LlmInsights | null;
}

export interface PipelineResultResponse {
  final_result: FinalResult;
  trace_data: TraceNode | null;
  stages_completed: PipelineStage[];
  stages?: PipelineStage[];
  updated_at: string;
  llm_insights?: LlmInsights | null;
}

export interface RunInput {
  file: File;
  targetColumn: string;
  predictionGoal: string;
  forceFailEval: boolean;
}

export interface HistoricalRun {
  runtimeSeconds: number;
  tokenCount: number;
  metricValue: number | null;
  retries: number;
  success: boolean;
  completedAt: string;
}

export interface ContractDefinition {
  name: string;
  description: string;
  fields: Record<string, string>;
}
