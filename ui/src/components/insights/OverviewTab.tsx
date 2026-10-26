import { CheckCircle2, Lightbulb, XCircle } from "lucide-react";
import type { ResultPayload } from "../../types";
import ImportanceChart from "./ImportanceChart";
import { Card, EmptyHint, Stat, fmt, prettyName } from "./shared";
import PerformanceTab from "./PerformanceTab";

export default function OverviewTab({ payload }: { payload: ResultPayload }) {
  const final = payload.final_result;
  const insights = payload.insights ?? {};
  const perf = insights.performance;
  const findings = insights.findings ?? [];
  const passed = final.status === "SUCCESS";
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Card title="What we found" subtitle="Plain-language summary of this run">
          {findings.length ? (
            <ul className="space-y-2.5">
              {findings.map((note) => (
                <li key={note} className="flex gap-2.5 text-sm leading-relaxed text-slate-300">
                  <Lightbulb size={16} className="mt-0.5 shrink-0 text-amber-300" />
                  <span>{note}</span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyHint>No findings were generated for this run.</EmptyHint>
          )}
          {final.llm_summary && (
            <p className="mt-4 rounded-lg border border-violet-500/20 bg-violet-500/[.06] p-3 text-sm leading-relaxed text-violet-100/80">
              {final.llm_summary}
            </p>
          )}
        </Card>
        <Card title="Model quality" subtitle="Measured on data the model never saw">
          <div className="grid grid-cols-2 gap-2">
            <Stat
              label="Verdict"
              value={<span className="inline-flex items-center gap-1.5">{passed ? <CheckCircle2 size={16} /> : <XCircle size={16} />}{passed ? "Ready to use" : "Below target"}</span>}
              tone={passed ? "emerald" : "rose"}
            />
            <Stat label="Best model" value={prettyName(final.model_name)} tone="cyan" />
            <Stat label={prettyName(final.metric_name)} value={fmt(final.metric_value)} hint={`target ${fmt(final.threshold)}`} />
            {perf?.kind === "regression" && <Stat label="Avg. error (MAE)" value={fmt(perf.mae)} hint={`${fmt(perf.within_10_percent)}% within 10%`} />}
            {perf?.kind === "classification" && <Stat label="Accuracy" value={`${fmt(perf.accuracy * 100)}%`} hint={`${perf.labels.length} classes`} />}
          </div>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="What drives the predictions" subtitle="How much accuracy drops when each input is shuffled">
          <ImportanceChart importance={insights.importance} />
        </Card>
        <PerformanceTab payload={payload} compact />
      </div>
    </div>
  );
}
