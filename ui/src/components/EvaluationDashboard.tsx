import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  Bot,
  CheckCircle2,
  Clock3,
  Download,
  ExternalLink,
  PackageCheck,
  RefreshCw,
  ShieldCheck,
  Workflow,
} from "lucide-react";
import { evaluationDownloadUrl, getEvaluationRun, getLatestEvaluationSuite, listEvaluationRuns } from "../api";
import type { EvaluationEvent, EvaluationRun, EvaluationSuite } from "../types";

const number = (value: unknown, digits = 3) => typeof value === "number" ? value.toFixed(digits) : "—";
const duration = (value: unknown) => typeof value === "number" ? value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms` : "—";
const pct = (value: number, total: number) => total ? `${((value / total) * 100).toFixed(0)}%` : "—";
const quantile = (values: number[], fraction: number) => {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
};

function eventStatus(status: EvaluationEvent["status"]) {
  if (status === "success" || status === "passed") return "text-emerald-300 bg-emerald-500/10 border-emerald-500/25";
  if (status === "retry") return "text-amber-300 bg-amber-500/10 border-amber-500/25";
  if (status === "running") return "text-sky-300 bg-sky-500/10 border-sky-500/25";
  return "text-rose-300 bg-rose-500/10 border-rose-500/25";
}

function Card({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Activity }) {
  return <article className="metric-card industrial-card">
    <div className="flex items-start justify-between gap-3">
      <div><p className="eyebrow">{label}</p><p className="mt-3 text-2xl font-semibold tracking-tight text-slate-50">{value}</p></div>
      <Icon size={18} className="text-sky-300" />
    </div>
    <p className="mt-3 text-xs text-slate-500">{note}</p>
  </article>;
}

function EventTable({ title, events }: { title: string; events: EvaluationEvent[] }) {
  return <section className="panel overflow-hidden">
    <header className="panel-header border-b border-slate-800"><div><h2 className="panel-title">{title}</h2><p className="panel-subtitle">Sanitized execution evidence</p></div><span className="font-mono text-xs text-slate-500">{events.length}</span></header>
    {events.length === 0 ? <p className="p-4 text-sm text-slate-500">No events recorded for this run.</p> : <div className="overflow-x-auto"><table className="evidence-table"><thead><tr><th>Event</th><th>Status</th><th>Duration</th><th>Source → destination</th><th>Detail</th></tr></thead><tbody>{events.map((event, index) => <tr key={`${event.name}-${index}`}><td className="font-medium text-slate-200">{event.name}</td><td><span className={`inline-flex rounded border px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ${eventStatus(event.status)}`}>{event.status}</span></td><td className="font-mono text-slate-400">{duration(event.duration_ms)}</td><td className="text-slate-400">{event.source || "—"}{event.destination ? ` → ${event.destination}` : ""}</td><td className="max-w-[320px] truncate text-slate-500" title={event.detail || ""}>{event.detail || "—"}</td></tr>)}</tbody></table></div>}
  </section>;
}

export default function EvaluationDashboard() {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selected, setSelected] = useState<EvaluationRun | null>(null);
  const [suite, setSuite] = useState<EvaluationSuite | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const [data, latestSuite] = await Promise.all([listEvaluationRuns(), getLatestEvaluationSuite()]);
      setRuns(data);
      setSuite(latestSuite);
      const fromUrl = new URLSearchParams(window.location.search).get("run_id");
      const current = fromUrl ? await getEvaluationRun(fromUrl).catch(() => null) : data[0] ?? null;
      setSelected(current);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Evaluation evidence is unavailable."); }
    finally { setLoading(false); }
  };
  useEffect(() => { void load(); }, []);
  const events = selected?.events ?? [];
  const agentEvents = events.filter((event) => event.category === "agent");
  const a2aEvents = events.filter((event) => event.category === "a2a");
  const mcpEvents = events.filter((event) => event.category === "mcp");
  const callbackEvents = events.filter((event) => event.category === "callback");
  const guardrailEvents = events.filter((event) => event.category === "guardrail");
  const successful = events.filter((event) => event.status === "success" || event.status === "passed").length;
  const metricEntries = useMemo(() => Object.entries(selected?.model_quality ?? {}), [selected]);
  const runtimes = runs.map((run) => Number(run.pipeline.runtime_ms)).filter((value) => Number.isFinite(value) && value >= 0);
  const p50 = quantile(runtimes, 0.5);
  const p95 = quantile(runtimes, 0.95);
  const metadataRows: Array<[string, unknown]> = selected ? [
    ["Dataset", selected.dataset.name], ["Rows", selected.dataset.rows], ["Target", selected.dataset.target_column],
    ["Model", selected.environment.model], ["Cache", selected.environment.llm_cache], ["Tokens", selected.cost.total_tokens],
    ["LLM cost", typeof selected.cost.estimated_llm_cost_usd === "number" ? `$${Number(selected.cost.estimated_llm_cost_usd).toFixed(6)}` : "—"],
    ["Events", `${successful}/${events.length} successful`],
  ] : [];
  const choose = async (run: EvaluationRun) => {
    const detail = await getEvaluationRun(run.run_id); setSelected(detail);
    window.history.replaceState({}, "", `/evaluation?run_id=${encodeURIComponent(run.run_id)}`);
  };

  return <div className="min-h-screen bg-canvas text-slate-200">
    <header className="sticky top-0 z-30 border-b border-slate-800 bg-canvas/95 backdrop-blur-xl"><div className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-3 px-4 py-3 lg:px-6"><div className="flex items-center gap-3"><div className="logo-mark"><Activity size={17} /></div><div><h1 className="text-base font-semibold tracking-tight text-slate-50">Evaluation Evidence</h1><p className="text-xs text-slate-500">Cloud execution, model quality and agentic-system observability</p></div></div><div className="flex items-center gap-2"><a href="/" className="action-link !w-auto px-3 text-xs"><ArrowLeft size={13} /> Operations</a><button type="button" onClick={() => void load()} className="action-link !w-auto px-3 text-xs"><RefreshCw size={13} className={loading ? "animate-spin" : ""} /> Refresh</button></div></div></header>
    <main className="mx-auto max-w-[1680px] space-y-4 px-4 py-5 lg:px-6">
      {error && <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">{error}</div>}
      {!loading && runs.length === 0 && <section className="panel p-8 text-center"><PackageCheck className="mx-auto mb-3 text-slate-500" /><h2 className="text-lg font-semibold text-slate-100">No retained evaluation evidence yet</h2><p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-slate-500">Complete a pipeline run after configuring Cloud Storage. This page intentionally shows retained evidence rather than an unverified static report.</p></section>}
      {selected && <>
        <section className="panel overflow-hidden"><div className="flex flex-wrap items-start justify-between gap-4 p-5"><div><p className="eyebrow">Official evaluation record</p><h2 className="mt-1 text-xl font-semibold text-slate-50">Run {selected.run_id.slice(0, 8)}</h2><p className="mt-1 text-sm text-slate-500">{new Date(selected.generated_at).toLocaleString()} · {selected.environment.deployment as string ?? "local"} · revision {selected.environment.revision as string ?? "unknown"}</p></div><div className="flex flex-wrap gap-2"><a className="action-link !w-auto px-3 text-xs" href={evaluationDownloadUrl(selected.run_id)}><Download size={13} /> Evidence JSON</a>{typeof selected.observability.langfuse_trace_url === "string" && <a className="action-link !w-auto px-3 text-xs" href={selected.observability.langfuse_trace_url} target="_blank" rel="noreferrer">Langfuse <ExternalLink size={13} /></a>}<span className={`status-chip ${selected.status === "SUCCESS" ? "status-success" : "status-failed"}`}>{selected.status}</span></div></div></section>
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6"><Card label={selected.primary_metric || "Model metric"} value={number(selected.primary_metric_value)} note={`Threshold ${number(selected.primary_threshold)}`} icon={CheckCircle2} /><Card label="Pipeline runtime" value={duration(selected.pipeline.runtime_ms)} note={`${selected.pipeline.successful_stages as number ?? 0}/${selected.pipeline.stage_count as number ?? 0} stages succeeded`} icon={Clock3} /><Card label="Latency p50 / p95" value={`${duration(p50)} / ${duration(p95)}`} note={`${runtimes.length} retained runs`} icon={Activity} /><Card label="Agent scenario suite" value={suite ? pct(suite.scenarios.filter((scenario) => scenario.passed).length, suite.scenarios.length) : "—"} note={suite ? `${suite.scenarios.length} Cloud Run Job scenarios` : "No formal suite retained"} icon={Bot} /><Card label="A2A reliability" value={pct(a2aEvents.filter((event) => event.status === "success").length, a2aEvents.length)} note={`${a2aEvents.length} JSON-RPC exchanges`} icon={Workflow} /><Card label="MCP reliability" value={pct(mcpEvents.filter((event) => event.status === "success").length, mcpEvents.length)} note={`${mcpEvents.length} validated tool calls`} icon={ShieldCheck} /></section>
        <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]"><div className="space-y-4"><section className="panel p-5"><div className="flex items-center justify-between"><div><h2 className="panel-title">Model quality</h2><p className="panel-subtitle">Held-out test-set metrics — not agent accuracy</p></div><span className="font-mono text-xs text-slate-500">{selected.dataset.task_type as string ?? "unknown"}</span></div><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{metricEntries.map(([key, value]) => <div key={key} className="rounded-md border border-slate-800 bg-slate-950/40 p-3"><p className="eyebrow">{key}</p><p className="mt-2 font-mono text-lg text-slate-100">{number(value)}</p></div>)}</div></section>{suite && <section className="panel p-5"><div className="flex items-center justify-between"><div><h2 className="panel-title">Formal agentic-system evaluation</h2><p className="panel-subtitle">Cloud Run Job · expected versus observed scenarios</p></div><span className="font-mono text-xs text-slate-500">{suite.suite_id}</span></div><div className="mt-4 grid gap-2 md:grid-cols-2">{suite.scenarios.map((scenario) => <div key={scenario.name} className="rounded-md border border-slate-800 bg-slate-950/40 p-3"><div className="flex justify-between gap-2"><strong className="text-sm text-slate-200">{scenario.name.replaceAll("_", " ")}</strong><span className={scenario.passed ? "text-emerald-300" : "text-rose-300"}>{scenario.passed ? "passed" : "failed"}</span></div><p className="mt-1 text-xs text-slate-500">Expected: {scenario.expected}</p><p className="mt-1 text-xs text-slate-400">Observed: {scenario.observed}</p></div>)}</div></section>}<EventTable title="Pipeline and agent trajectory" events={events.filter((event) => event.category === "pipeline" || event.category === "agent")} /><EventTable title="A2A communication" events={a2aEvents} /><EventTable title="MCP tools" events={mcpEvents} /><EventTable title="Guardrails" events={guardrailEvents} /><EventTable title="Callbacks and validation" events={callbackEvents} /></div>
          <aside className="space-y-4"><section className="panel p-4"><h2 className="panel-title">Run metadata</h2><dl className="mt-3 space-y-3 text-xs">{metadataRows.map(([label, value]) => <div key={label} className="flex justify-between gap-3 border-b border-slate-800 pb-2 last:border-0"><dt className="text-slate-500">{label}</dt><dd className="max-w-[190px] truncate text-right font-mono text-slate-300" title={String(value ?? "—")}>{String(value ?? "—")}</dd></div>)}</dl></section><section className="panel p-4"><h2 className="panel-title">Retained runs</h2><div className="mt-3 space-y-2">{runs.map((run) => <button key={run.run_id} type="button" onClick={() => void choose(run)} className={`w-full rounded-md border p-3 text-left transition ${run.run_id === selected.run_id ? "border-sky-500/50 bg-sky-500/10" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"}`}><div className="flex justify-between gap-2"><span className="font-mono text-xs text-slate-200">{run.run_id.slice(0, 8)}</span><span className="text-[10px] text-slate-500">{run.status}</span></div><p className="mt-1 truncate text-xs text-slate-500">{String(run.dataset.name ?? "dataset")} · {new Date(run.generated_at).toLocaleDateString()}</p></button>)}</div></section></aside>
        </section>
      </>}
    </main>
  </div>;
}
