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
  prediction_goal?: string;
  target_column?: string;
  task_type?: string;
  endpoint?: string | null;
  deployment?: DeploymentInfo | null;
  deployment_error?: string | null;
  model_card_markdown?: string;
}

export interface DeploymentInfo {
  run_id: string;
  image_digest: string;
  service_url: string;
  health_status: string;
  deployment_revision: string;
  deployed_at: string;
}

export interface UsageAgentBucket {
  total_tokens: number;
  requests: number;
  cache_hits?: number;
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
  cache_hits?: number;
  local_llm_requests?: number;
  fallback_decisions?: number;
  calls?: UsageCall[];
}

export interface UsageCall {
  stage: string;
  agent: string;
  framework: string;
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  requests: number;
  latency_seconds: number;
  estimated_cost_usd: number;
  source: string;
}

export interface Candidate {
  name: string;
  mlflow_run_id?: string;
  parameters: Record<string, JsonValue>;
  metrics: Record<string, number>;
  primary_metric: string;
  training_time_ms: number;
  eligible: boolean;
  selected: boolean;
}

export interface Attempt {
  attempt: number;
  retry_feedback_in: string | null;
  selected_model: string;
  candidates: Candidate[];
  reasoning: string;
  evaluation: {
    passed: boolean;
    primary_metric: string;
    held_out_metrics: Record<string, number>;
    threshold: number;
    direction: string;
    feedback: string | null;
    reasoning: string;
  } | null;
}

export interface DatasetInsights {
  rows: number;
  columns: number;
  numeric_features: string[];
  categorical_features: string[];
  missing_cells: number;
  duplicate_rows: number;
  missing_by_column: { column: string; missing: number; percent: number }[];
  target_column: string;
  task_type: string;
  target_distribution: { bin?: string; label?: string; count: number; percent?: number }[];
  target_summary: Record<string, number | null>;
  relationships: { feature: string; correlation: number }[];
  dataset_hash: string | null;
  quality_findings: string[];
  train_rows: number | null;
  test_rows: number | null;
  used_features: string[];
}

export interface RegressionPerformance {
  kind: "regression";
  scatter: { actual: number; predicted: number }[];
  residual_histogram: { bin: string; count: number }[];
  mae: number;
  rmse: number;
  median_abs_error: number;
  within_10_percent: number;
}

export interface ClassificationPerformance {
  kind: "classification";
  labels: string[];
  confusion_matrix: number[][];
  per_class: { label: string; precision: number; recall: number; f1: number; support: number }[];
  accuracy: number;
  roc?: { fpr: number; tpr: number }[];
  positive_label?: string;
}

export interface Insights {
  dataset?: DatasetInsights;
  performance?: RegressionPerformance | ClassificationPerformance;
  importance?: { feature: string; importance: number; std: number; share_percent: number }[];
  attempts?: Attempt[];
  findings?: string[];
  model_card_markdown?: string;
  errors?: string[];
}

export interface A2AExchange {
  attempt: number;
  method: string;
  agent: string;
  endpoint: string;
  protocol_version: string;
  task_id: string;
  task_state: string;
  latency_ms: number;
  request: Record<string, JsonValue>;
  response: Record<string, JsonValue>;
}

export interface A2ALog {
  agent_card: Record<string, JsonValue>;
  exchanges: A2AExchange[];
}

export interface RunInputMeta {
  dataset_name?: string;
  target_column?: string;
  prediction_goal?: string;
  force_fail_eval?: boolean;
}

export interface ResultPayload {
  final_result: FinalResult;
  trace_data: TraceNode | null;
  llm_insights?: LlmInsights | null;
  insights?: Insights | null;
  a2a?: A2ALog | null;
  run_input?: RunInputMeta | null;
  stages_completed?: PipelineStage[];
  stages?: PipelineStage[];
  updated_at?: string;
}

export interface ProviderState {
  role: string;
  model: string;
  provider: string;
  ok: boolean;
  detail: string;
  credit_usd?: number | null;
}

export interface ProvidersResponse {
  mode: string;
  providers: ProviderState[];
  cache: { enabled: boolean; entries: number };
  checked_at: string;
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  status: string;
  dataset_name: string | null;
  target_column: string | null;
  prediction_goal: string | null;
  task_type: string | null;
  model_name: string | null;
  metric_name: string | null;
  metric_value: number | null;
  runtime_seconds: number;
  cost_usd: number;
  tokens: number;
  llm_calls: number;
  cache_hits: number;
  retries: number;
}

export interface SampleDataset {
  name: string;
  title: string;
  target_column: string;
  prediction_goal: string;
}

export interface FeatureSpec {
  name: string;
  type: "number" | "category";
  example: number | string;
  min?: number | null;
  max?: number | null;
  categories?: string[];
}

export interface DeploymentMetadata {
  run_id: string;
  task_type: string;
  target_column: string;
  model_name: string;
  features: FeatureSpec[];
  held_out_metrics?: Record<string, number>;
  primary_metric?: string;
  auth_required?: boolean;
}

export interface Prediction {
  prediction: string | number;
  probabilities?: Record<string, number>;
}

export interface GuardrailCheck {
  allowed: boolean;
  category: string | null;
  reason: string;
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

export type PipelineResultResponse = ResultPayload;

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
