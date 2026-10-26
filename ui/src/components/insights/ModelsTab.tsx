import { ArrowDown, CheckCircle2, Crown, MessageSquareWarning, XCircle } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Attempt } from "../../types";
import { AXIS_PROPS, COLORS, Card, EmptyHint, TOOLTIP_PROPS, fmt, prettyName } from "./shared";

// Fixed by the pipeline (not chosen by the LLM), so not shown as "settings".
const HIDDEN_PARAMS = new Set(["model_type", "random_state", "retry_attempt"]);

function AttemptCard({ attempt, total }: { attempt: Attempt; total: number }) {
  const metric = attempt.candidates[0]?.primary_metric ?? "metric";
  const rows = attempt.candidates.map((candidate) => ({
    name: prettyName(candidate.name),
    value: candidate.metrics[metric],
    time: candidate.training_time_ms,
    selected: candidate.selected,
    eligible: candidate.eligible,
  }));
  const evaluation = attempt.evaluation;
  return (
    <Card
      title={`Attempt ${attempt.attempt} of ${total}`}
      subtitle={`Validation ${metric} for each candidate · picked: ${prettyName(attempt.selected_model)}`}
      action={evaluation && (
        <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${evaluation.passed ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" : "border-rose-500/30 bg-rose-500/10 text-rose-300"}`}>
          {evaluation.passed ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
          held-out {fmt(evaluation.held_out_metrics[evaluation.primary_metric])} vs {fmt(evaluation.threshold)}
        </span>
      )}
    >
      {attempt.retry_feedback_in && (
        <p className="mb-3 flex gap-2 rounded-lg border border-amber-500/20 bg-amber-500/[.06] p-2.5 text-xs leading-relaxed text-amber-100/80">
          <MessageSquareWarning size={15} className="mt-0.5 shrink-0 text-amber-300" />
          <span><strong className="text-amber-200">Feedback from the evaluator: </strong>{attempt.retry_feedback_in}</span>
        </p>
      )}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ left: 0, right: 8, top: 20, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke={COLORS.grid} />
              <XAxis dataKey="name" {...AXIS_PROPS} />
              <YAxis {...AXIS_PROPS} tickFormatter={(v: number) => fmt(v)} width={60} domain={[0, "auto"]} />
              <Tooltip {...TOOLTIP_PROPS} formatter={(value: number) => [fmt(value), metric]} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {rows.map((row) => (
                  <Cell key={row.name} fill={row.selected ? COLORS.emerald : row.eligible ? COLORS.cyan : "#334155"} />
                ))}
                <LabelList dataKey="value" position="top" formatter={(v: number) => fmt(v)} fill="#cbd5e1" fontSize={11} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-1 text-center text-[11px] text-slate-500">
            <span className="text-emerald-300">■</span> chosen · <span className="text-cyan-300">■</span> within 1% quality band · <span className="text-slate-500">■</span> outside band
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500">
                <th className="py-2 pr-2 font-medium">Model</th>
                <th className="py-2 pr-2 font-medium">Settings chosen by the LLM</th>
                <th className="py-2 font-medium text-right">Train time</th>
              </tr>
            </thead>
            <tbody>
              {attempt.candidates.map((candidate) => (
                <tr key={candidate.name} className={`border-b border-slate-800/60 ${candidate.selected ? "bg-emerald-500/[.05]" : ""}`}>
                  <td className="py-2 pr-2 font-medium text-slate-200">
                    <span className="inline-flex items-center gap-1">{candidate.selected && <Crown size={12} className="text-emerald-300" />}{prettyName(candidate.name)}</span>
                  </td>
                  <td className="py-2 pr-2 font-mono text-[11px] text-slate-400">
                    {Object.keys(candidate.parameters).length
                      ? Object.entries(candidate.parameters).filter(([key]) => !HIDDEN_PARAMS.has(key)).map(([key, value]) => `${key}=${value ?? "none"}`).join(", ") || "defaults"
                      : "defaults"}
                  </td>
                  <td className="py-2 text-right font-mono text-slate-400">{candidate.training_time_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {attempt.reasoning && (
        <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <summary className="cursor-pointer text-xs font-medium text-slate-400">Why the ADK agent chose these settings and this model</summary>
          <p className="mt-2 whitespace-pre-line text-xs leading-relaxed text-slate-300">{attempt.reasoning}</p>
        </details>
      )}
    </Card>
  );
}

export default function ModelsTab({ attempts }: { attempts?: Attempt[] }) {
  if (!attempts?.length) return <EmptyHint>Model comparisons appear after the model-selection agent has run.</EmptyHint>;
  return (
    <div className="space-y-2">
      {attempts.map((attempt, index) => (
        <div key={attempt.attempt}>
          <AttemptCard attempt={attempt} total={attempts.length} />
          {index < attempts.length - 1 && (
            <div className="flex items-center justify-center gap-2 py-2 text-xs text-amber-300">
              <ArrowDown size={14} /> evaluator sent feedback over A2A — retrying
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
