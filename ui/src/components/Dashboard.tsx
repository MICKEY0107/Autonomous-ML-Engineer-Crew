import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, BrainCircuit, Circle, Database, Radio, ShieldCheck, Workflow } from "lucide-react";
import { apiErrorMessage, getPipelineResult, getPipelineStatus, startPipeline } from "../api";
import type {
  FinalResult,
  HistoricalRun,
  PipelineStatusResponse,
  RunInput,
  TraceNode,
} from "../types";
import InputPanel from "./InputPanel";
import MetricCard from "./MetricCard";
import PipelineTimeline from "./PipelineTimeline";
import ResultTabs from "./ResultTabs";
import ReasoningPanel from "./ReasoningPanel";

const INITIAL_STAGES = [
  "Input Validation",
  "Data-Prep Agent",
  "MCP: profile_dataset",
  "Model-Selection Agent",
  "A2A: message/send",
  "MCP: log_experiment",
  "Evaluation Agent",
  "Retry Controller",
  "Terminal Result",
].map((name) => ({ name, duration_ms: 0, status: "pending" as const, detail: "Awaiting pipeline run", attempts: 0 }));

const EMPTY_STATUS: PipelineStatusResponse = {
  status: "idle",
  current_stage: null,
  retry_count: 0,
  stages_completed: INITIAL_STAGES,
  final_result: null,
  trace_data: null,
  error: null,
  started_at: null,
  updated_at: new Date().toISOString(),
};

const HISTORY_KEY = "amec-dashboard-history-v1";

function loadHistory(): HistoricalRun[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.slice(-5) : [];
  } catch {
    return [];
  }
}

function percentTrend(values: number[]): number | null {
  if (values.length < 2) return null;
  const previous = values.at(-2) ?? 0;
  const current = values.at(-1) ?? 0;
  return previous === 0 ? 0 : ((current - previous) / Math.abs(previous)) * 100;
}

function statusPresentation(status: PipelineStatusResponse["status"]) {
  if (status === "running") return { label: "Running", tone: "status-running" };
  if (status === "success") return { label: "Success", tone: "status-success" };
  if (status === "failed") return { label: "Failed", tone: "status-failed" };
  return { label: "Idle", tone: "status-idle" };
}

function formatMetric(value: number): string {
  if (Math.abs(value) >= 1_000) {
    return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(value);
  }
  return value.toFixed(3);
}

export default function Dashboard() {
  const [pipeline, setPipeline] = useState<PipelineStatusResponse>(EMPTY_STATUS);
  const [result, setResult] = useState<FinalResult | null>(null);
  const [trace, setTrace] = useState<TraceNode | null>(null);
  const [history, setHistory] = useState<HistoricalRun[]>(loadHistory);
  const [error, setError] = useState<string | null>(null);
  const resultFetched = useRef(false);

  const persistResult = useCallback((finalResult: FinalResult) => {
    setHistory((current) => {
      if (current.some((item) => item.completedAt === finalResult.completed_at)) return current;
      const next = [...current, {
        runtimeSeconds: finalResult.runtime_ms / 1000,
        tokenCount: finalResult.token_count,
        metricValue: finalResult.metric_value,
        retries: finalResult.retry_count,
        success: finalResult.status === "SUCCESS",
        completedAt: finalResult.completed_at,
      }].slice(-5);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const hydrateFromBackend = async () => {
      try {
        const current = await getPipelineStatus();
        if (cancelled) return;
        setPipeline(current);
        if ((current.status === "success" || current.status === "failed") && current.final_result) {
          const completed = await getPipelineResult();
          if (cancelled) return;
          resultFetched.current = true;
          setResult(completed.final_result);
          setTrace(completed.trace_data);
          persistResult(completed.final_result);
        } else if (current.status === "running") {
          resultFetched.current = false;
        }
      } catch (requestError) {
        if (!cancelled) setError(`Backend unavailable: ${apiErrorMessage(requestError)}`);
      }
    };
    void hydrateFromBackend();
    return () => { cancelled = true; };
  }, [persistResult]);

  useEffect(() => {
    if (pipeline.status !== "running") return undefined;
    const poll = window.setInterval(async () => {
      try {
        const next = await getPipelineStatus();
        setPipeline(next);
        if (next.status !== "running" && !resultFetched.current) {
          resultFetched.current = true;
          const completed = await getPipelineResult();
          setResult(completed.final_result);
          setTrace(completed.trace_data);
          persistResult(completed.final_result);
        }
      } catch (requestError) {
        setError(apiErrorMessage(requestError));
      }
    }, 500);
    return () => window.clearInterval(poll);
  }, [persistResult, pipeline.status]);

  const handleRun = useCallback(async (input: RunInput) => {
    setError(null);
    setResult(null);
    setTrace(null);
    resultFetched.current = false;
    try {
      const started = await startPipeline(input);
      setPipeline(started);
    } catch (requestError) {
      setError(apiErrorMessage(requestError));
    }
  }, []);

  const runtimeValues = history.map((item) => item.runtimeSeconds);
  const tokenValues = history.map((item) => item.tokenCount);
  const metricValues = history.flatMap((item) => item.metricValue == null ? [] : [item.metricValue]);
  const retryValues = history.map((item) => item.retries);
  const successRate = history.length ? history.filter((item) => item.success).length / history.length * 100 : 0;
  const liveRuntime = pipeline.status === "running" && pipeline.started_at
    ? Math.max(0, (Date.now() - Date.parse(pipeline.started_at)) / 1000)
    : result?.runtime_ms != null ? result.runtime_ms / 1000 : runtimeValues.at(-1) ?? 0;
  const currentMetric = result?.metric_value ?? metricValues.at(-1) ?? 0;
  const lastUpdated = useMemo(() => new Intl.DateTimeFormat(undefined, {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  }).format(new Date(pipeline.updated_at)), [pipeline.updated_at]);
  const status = statusPresentation(pipeline.status);
  const safeData = (values: number[]) => values.length ? values : [0, 0, 0, 0, 0];
  const reasoningStages = pipeline.stages_completed.filter(
    (stage) => stage.reasoning && Object.keys(stage.reasoning).length > 0,
  );

  return (
    <div className="min-h-screen bg-canvas text-slate-200">
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-canvas/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-4 py-3 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="logo-mark"><Bot size={17} /></div>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold tracking-tight text-slate-50">Autonomous ML Engineer Crew</h1>
              <p className="mt-0.5 flex items-center gap-1.5 text-[9px] uppercase tracking-[.18em] text-slate-600">
                <Workflow size={9} /> CrewAI control plane <span className="text-slate-800">/</span> Google ADK A2A
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden items-center gap-1.5 rounded-md border border-slate-800 bg-slate-950/50 px-2 py-1.5 text-[9px] uppercase tracking-wider text-slate-500 sm:inline-flex">
              <ShieldCheck size={11} /> row-safe telemetry
            </span>
            <span className={`status-chip ${status.tone}`}><Circle size={7} fill="currentColor" /> {status.label}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] space-y-4 px-4 py-4 lg:px-6">
        <section className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <MetricCard value={`${liveRuntime.toFixed(1)}s`} label="Runtime" trend={percentTrend(runtimeValues)} color="#06b6d4" sparkline={safeData(runtimeValues)} footnote={pipeline.status === "running" ? "live elapsed time" : "vs previous run"} />
          <MetricCard value={(result?.token_count ?? tokenValues.at(-1) ?? 0).toLocaleString()} label="Token Count" trend={percentTrend(tokenValues)} color="#8b5cf6" sparkline={safeData(tokenValues)} />
          <MetricCard value={result?.metric_value == null && !history.length ? "—" : formatMetric(currentMetric)} label="Model Accuracy / R²" trend={percentTrend(metricValues)} color="#10b981" sparkline={safeData(metricValues)} chart="bar" footnote={result?.metric_name ?? "held-out metric"} />
          <MetricCard value={String(pipeline.retry_count)} label="Retry Count" trend={percentTrend(retryValues)} color={pipeline.retry_count ? "#f59e0b" : "#10b981"} sparkline={safeData(retryValues)} chart="bar" footnote="bounded at two" />
          <MetricCard value={`${successRate.toFixed(0)}%`} label="Pipeline Success Rate" trend={null} color="#10b981" sparkline={[]} chart="progress" progress={successRate} />
          <MetricCard value={result?.estimated_cost_usd != null ? `$${result.estimated_cost_usd.toFixed(5)}` : "—"} label="Est. LLM Cost" trend={null} color="#64748b" sparkline={[]} chart="none" footnote={`${result?.usage_report?.llm_requests ?? 0} LLM calls · updated ${lastUpdated}`} />
        </section>

        <section className="grid items-start gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(360px,2fr)]">
          <PipelineTimeline stages={pipeline.stages_completed} retryCount={pipeline.retry_count} />
          <ResultTabs result={result} trace={trace} />
        </section>

        {reasoningStages.length > 0 && (
          <section className="panel overflow-hidden" aria-labelledby="reasoning-title">
            <div className="panel-header border-b border-slate-800/80">
              <div>
                <h2 id="reasoning-title" className="panel-title flex items-center gap-2">
                  <BrainCircuit size={13} className="text-cyan-400" /> Decision reasoning
                </h2>
                <p className="panel-subtitle">Live, row-safe evidence behind each autonomous decision</p>
              </div>
              <span className="eyebrow">{reasoningStages.length} stages explained</span>
            </div>
            <div className="grid gap-2 p-3 lg:grid-cols-2">
              {reasoningStages.map((stage, index) => (
                <ReasoningPanel
                  key={stage.name}
                  stage={stage}
                  defaultOpen={index === reasoningStages.length - 1 || stage.status === "failed" || stage.status === "retry"}
                />
              ))}
            </div>
          </section>
        )}

        {error && (
          <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-[11px] text-rose-300" role="alert">{error}</div>
        )}
        <InputPanel isRunning={pipeline.status === "running"} statusLabel={pipeline.status === "running" ? "Running" : result ? "Complete" : "Ready"} onRun={handleRun} />

        <footer className="flex flex-col gap-2 py-3 text-[9px] uppercase tracking-[.14em] text-slate-700 sm:flex-row sm:items-center sm:justify-between">
          <span className="inline-flex items-center gap-1.5"><Database size={10} /> MLflow experiments · Langfuse traces · Pydantic v2 contracts</span>
          <span className="inline-flex items-center gap-1.5"><Radio size={10} /> API polling interval 500ms</span>
        </footer>
      </main>
    </div>
  );
}
