import type { ReactNode } from "react";
import { CheckCircle2, Download, ExternalLink, FileText, Rocket, XCircle } from "lucide-react";
import { downloadUrl } from "../api";
import type { ResultPayload } from "../types";
import { fmt, prettyName } from "./insights/shared";

export default function ResultSummary({ payload }: { payload: ResultPayload }) {
  const final = payload.final_result;
  const usage = final.usage_report ?? {};
  const success = final.status === "SUCCESS";
  const inputFailure = final.status === "FAILED_INPUT";
  const headline = payload.insights?.findings?.[1] ?? payload.insights?.findings?.[0];
  return (
    <section className={`rounded-xl border p-4 ${success ? "border-emerald-500/30 bg-emerald-500/[.06]" : "border-rose-500/30 bg-rose-500/[.06]"}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className={`flex items-center gap-2 text-lg font-semibold ${success ? "text-emerald-200" : "text-rose-200"}`}>
            {success ? <CheckCircle2 size={20} /> : <XCircle size={20} />}
            {success
              ? `${prettyName(final.model_name)} is ready${final.deployment ? " and deployed" : ""}`
              : inputFailure ? "Input rejected by guardrails" : "No model met the quality target"}
          </div>
          <p className="mt-1 max-w-3xl text-sm text-slate-300">
            {inputFailure ? final.error : headline ?? final.llm_summary}
          </p>
          {payload.run_input && (
            <p className="mt-1 text-xs text-slate-500">
              {payload.run_input.dataset_name} · target <span className="text-slate-300">{payload.run_input.target_column}</span> · “{payload.run_input.prediction_goal}”
            </p>
          )}
        </div>
        {!inputFailure && (
          <div className="flex flex-wrap gap-2">
            <a className="action-link w-auto text-xs" href={downloadUrl(final.run_id, "model-card.md")}><FileText size={13} /> Model card</a>
            <a className="action-link w-auto text-xs" href={downloadUrl(final.run_id, "result.json")}><Download size={13} /> Result JSON</a>
            <a className="action-link w-auto text-xs" href={final.trace_url} target="_blank" rel="noreferrer"><ExternalLink size={13} /> Langfuse trace</a>
            <a className="action-link w-auto text-xs" href={final.mlflow_url} target="_blank" rel="noreferrer"><ExternalLink size={13} /> MLflow</a>
          </div>
        )}
      </div>
      {!inputFailure && (
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
          <Kpi label={prettyName(final.metric_name)} value={fmt(final.metric_value)} hint={`target ${fmt(final.threshold)}`} />
          <Kpi label="Attempts" value={String((final.retry_count ?? 0) + 1)} hint={`${final.retry_count} retries`} />
          <Kpi label="Runtime" value={`${(final.runtime_ms / 1000).toFixed(1)}s`} />
          <Kpi label="LLM calls" value={String(usage.llm_requests ?? 0)} hint={`${usage.local_llm_requests ?? 0} local · ${usage.cache_hits ?? 0} cached`} />
          <Kpi label="Tokens" value={(final.token_count ?? 0).toLocaleString()} />
          <Kpi label="LLM cost" value={`$${(final.estimated_cost_usd ?? 0).toFixed(5)}`} hint={final.deployment ? <span className="inline-flex items-center gap-1"><Rocket size={11} /> {final.deployment.health_status}</span> : undefined} />
        </div>
      )}
    </section>
  );
}

function Kpi({ label, value, hint }: { label: string; value: string; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-0.5 text-base font-semibold text-slate-100">{value}</div>
      {hint && <div className="text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}
