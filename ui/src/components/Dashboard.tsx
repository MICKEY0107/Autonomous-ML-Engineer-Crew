import { useCallback, useEffect, useRef, useState } from "react";
import {
  BarChart3,
  Bot,
  BrainCircuit,
  Circle,
  Database,
  GitBranch,
  History,
  LayoutDashboard,
  Loader2,
  Rocket,
  Sparkles,
  Table2,
  Users,
} from "lucide-react";
import { apiErrorMessage, getPipelineResult, getPipelineStatus, getRun, startPipeline } from "../api";
import type { PipelineStatusResponse, ResultPayload, RunInput } from "../types";
import HistoryPanel from "./HistoryPanel";
import InputPanel from "./InputPanel";
import PipelineTimeline from "./PipelineTimeline";
import ProviderBar from "./ProviderBar";
import ResultSummary from "./ResultSummary";
import ResultTabs from "./ResultTabs";
import AgentsTab from "./insights/AgentsTab";
import DataTab from "./insights/DataTab";
import ModelsTab from "./insights/ModelsTab";
import OverviewTab from "./insights/OverviewTab";
import PerformanceTab from "./insights/PerformanceTab";
import PredictTab from "./insights/PredictTab";

const INITIAL_STAGES = [
  "Input Validation",
  "Data-Prep Agent",
  "MCP: profile_dataset",
  "Model-Selection Agent",
  "A2A: message/send",
  "MCP: log_experiment",
  "Evaluation Agent",
  "Retry Controller",
  "Deployment Agent",
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

const TABS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "data", label: "Data", icon: Table2 },
  { id: "models", label: "Models", icon: BarChart3 },
  { id: "performance", label: "Performance", icon: Sparkles },
  { id: "agents", label: "Agents & A2A", icon: Users },
  { id: "predict", label: "Predict", icon: Rocket },
  { id: "trace", label: "Trace & contracts", icon: GitBranch },
] as const;
type TabId = (typeof TABS)[number]["id"];

const STAGE_HELP: Record<string, string> = {
  "Input Validation": "Checking the file, the target column and the goal against the guardrails",
  "Data-Prep Agent": "CrewAI agent profiles the data and decides how to clean and split it",
  "MCP: profile_dataset": "MCP tool profiling schema, missing values and leakage risks",
  "Model-Selection Agent": "Google ADK agent plans settings and trains three candidate models",
  "A2A: message/send": "CrewAI orchestrator talking to the ADK agent over the A2A protocol",
  "MCP: log_experiment": "MCP tool logging every candidate to MLflow",
  "Evaluation Agent": "CrewAI agent tests the winner on unseen data and reflects on the result",
  "Retry Controller": "Sending feedback back to the model-selection agent",
  "Deployment Agent": "Packaging the accepted model behind a prediction API",
  "Terminal Result": "Writing the report and insights",
};

function statusPresentation(status: PipelineStatusResponse["status"]) {
  if (status === "running") return { label: "Running", tone: "status-running" };
  if (status === "success") return { label: "Success", tone: "status-success" };
  if (status === "failed") return { label: "Failed", tone: "status-failed" };
  return { label: "Idle", tone: "status-idle" };
}

function Welcome() {
  const steps = [
    { icon: Database, title: "Upload a CSV", text: "Or pick a sample. Any tabular data with a header row." },
    { icon: Table2, title: "Pick what to predict", text: "Choose the target column and describe the goal in plain words." },
    { icon: Bot, title: "Agents do the rest", text: "They clean the data, train and compare models, test the winner and deploy it." },
  ];
  const outputs = [
    "A trained model that passed a held-out quality check",
    "Charts: what drives predictions, errors, class balance, model comparison",
    "A live prediction API and a form to try it",
    "A downloadable model card and a full audit trail of every agent decision",
  ];
  return (
    <section className="rounded-xl border border-slate-800 bg-panel/90 p-6">
      <h2 className="text-xl font-semibold text-slate-50">Automated machine learning, explained as it happens</h2>
      <p className="mt-1 text-sm text-slate-400">No ML expertise needed. A crew of AI agents builds, checks and ships a model for your data.</p>
      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {steps.map((step, index) => (
          <div key={step.title} className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
            <step.icon size={20} className="text-cyan-300" />
            <div className="mt-2 text-sm font-semibold text-slate-100">{index + 1}. {step.title}</div>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">{step.text}</p>
          </div>
        ))}
      </div>
      <div className="mt-5 rounded-lg border border-emerald-500/20 bg-emerald-500/[.05] p-4">
        <div className="text-sm font-semibold text-emerald-200">What you get</div>
        <ul className="mt-2 grid gap-1.5 text-sm text-slate-300 md:grid-cols-2">
          {outputs.map((item) => <li key={item} className="flex gap-2"><Sparkles size={14} className="mt-0.5 shrink-0 text-emerald-300" />{item}</li>)}
        </ul>
      </div>
    </section>
  );
}

function RunningCard({ pipeline }: { pipeline: PipelineStatusResponse }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, []);
  const elapsed = pipeline.started_at ? Math.max(0, (now - Date.parse(pipeline.started_at)) / 1000) : 0;
  const stage = pipeline.current_stage ?? "Starting";
  const done = pipeline.stages_completed.filter((item) => item.status === "success").length;
  return (
    <section className="rounded-xl border border-cyan-500/30 bg-cyan-500/[.05] p-6">
      <div className="flex items-center gap-3">
        <Loader2 size={22} className="animate-spin text-cyan-300" />
        <div>
          <div className="text-lg font-semibold text-cyan-100">{stage}</div>
          <p className="text-sm text-slate-400">{STAGE_HELP[stage] ?? "Agents are working"}</p>
        </div>
        <span className="ml-auto font-mono text-sm text-slate-300">{elapsed.toFixed(1)}s</span>
      </div>
      <div className="mt-4 h-2 rounded-full bg-slate-800">
        <div className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all" style={{ width: `${Math.min(100, (done / pipeline.stages_completed.length) * 100)}%` }} />
      </div>
      {pipeline.retry_count > 0 && (
        <p className="mt-3 text-sm text-amber-300">Retry {pipeline.retry_count}/2 — the evaluator sent feedback to the model-selection agent.</p>
      )}
    </section>
  );
}

export default function Dashboard() {
  const [pipeline, setPipeline] = useState<PipelineStatusResponse>(EMPTY_STATUS);
  const [payload, setPayload] = useState<ResultPayload | null>(null);
  const [tab, setTab] = useState<TabId>("overview");
  const [error, setError] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyKey, setHistoryKey] = useState(0);
  const resultFetched = useRef(false);

  const loadResult = useCallback(async () => {
    const completed = await getPipelineResult();
    setPayload(completed);
    setTab(completed.final_result.status === "FAILED_INPUT" ? "trace" : "overview");
    setHistoryKey((key) => key + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const current = await getPipelineStatus();
        if (cancelled) return;
        setPipeline(current);
        if ((current.status === "success" || current.status === "failed") && current.final_result) {
          resultFetched.current = true;
          await loadResult();
        }
      } catch (requestError) {
        if (!cancelled) setError(`Backend unavailable: ${apiErrorMessage(requestError)}`);
      }
    })();
    return () => { cancelled = true; };
  }, [loadResult]);

  useEffect(() => {
    if (pipeline.status !== "running") return undefined;
    const poll = window.setInterval(async () => {
      try {
        const next = await getPipelineStatus();
        setPipeline(next);
        if (next.status !== "running" && !resultFetched.current) {
          resultFetched.current = true;
          await loadResult();
        }
      } catch (requestError) {
        setError(apiErrorMessage(requestError));
      }
    }, 700);
    return () => window.clearInterval(poll);
  }, [loadResult, pipeline.status]);

  const handleRun = useCallback(async (input: RunInput) => {
    setError(null);
    setPayload(null);
    resultFetched.current = false;
    try {
      setPipeline(await startPipeline(input));
    } catch (requestError) {
      setError(apiErrorMessage(requestError));
    }
  }, []);

  const openRun = useCallback(async (runId: string) => {
    try {
      const past = await getRun(runId);
      setPayload(past);
      setTab("overview");
      if (past.stages_completed) setPipeline((current) => ({ ...current, status: past.final_result.status === "SUCCESS" ? "success" : "failed", stages_completed: past.stages_completed ?? current.stages_completed }));
    } catch (requestError) {
      setError(apiErrorMessage(requestError));
    }
  }, []);

  const status = statusPresentation(pipeline.status);
  const running = pipeline.status === "running";

  return (
    <div className="min-h-screen bg-canvas text-slate-200">
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-canvas/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-3 px-4 py-3 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="logo-mark"><BrainCircuit size={17} /></div>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight text-slate-50">Autonomous ML Engineer Crew</h1>
              <p className="text-xs text-slate-500">Dataset + target + goal → trained, explained and deployed model</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <ProviderBar />
            <button type="button" className="action-link w-auto text-xs" onClick={() => setHistoryOpen(true)}><History size={13} /> History</button>
            <span className={`status-chip ${status.tone}`}><Circle size={7} fill="currentColor" /> {status.label}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1680px] gap-4 px-4 py-4 lg:px-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        <aside className="space-y-4">
          <InputPanel isRunning={running} onRun={handleRun} />
          <PipelineTimeline stages={pipeline.stages_completed} retryCount={pipeline.retry_count} />
        </aside>

        <div className="min-w-0 space-y-4">
          {error && <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300" role="alert">{error}</div>}
          {running && <RunningCard pipeline={pipeline} />}
          {!running && !payload && <Welcome />}
          {!running && payload && (
            <>
              <ResultSummary payload={payload} />
              <nav className="flex gap-1 overflow-x-auto border-b border-slate-800">
                {TABS.map((item) => (
                  <button key={item.id} type="button" onClick={() => setTab(item.id)}
                    className={`inline-flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2.5 text-sm transition-colors ${tab === item.id ? "border-cyan-400 text-cyan-200" : "border-transparent text-slate-500 hover:text-slate-300"}`}>
                    <item.icon size={15} /> {item.label}
                  </button>
                ))}
              </nav>
              {tab === "overview" && <OverviewTab payload={payload} />}
              {tab === "data" && <DataTab data={payload.insights?.dataset} />}
              {tab === "models" && <ModelsTab attempts={payload.insights?.attempts} />}
              {tab === "performance" && <PerformanceTab payload={payload} />}
              {tab === "agents" && <AgentsTab payload={payload} />}
              {tab === "predict" && <PredictTab final={payload.final_result} />}
              {tab === "trace" && <ResultTabs result={payload.final_result} trace={payload.trace_data} />}
            </>
          )}
        </div>
      </main>
      <HistoryPanel open={historyOpen} onClose={() => setHistoryOpen(false)} onSelect={(runId) => void openRun(runId)} refreshKey={historyKey} />
    </div>
  );
}
