import { useEffect, useState } from "react";
import { History, X } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { listRuns } from "../api";
import type { RunSummary } from "../types";
import { AXIS_PROPS, COLORS, TOOLTIP_PROPS, fmt, prettyName } from "./insights/shared";

export default function HistoryPanel({ open, onClose, onSelect, refreshKey }: {
  open: boolean;
  onClose: () => void;
  onSelect: (runId: string) => void;
  refreshKey: number;
}) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  useEffect(() => {
    if (open) listRuns().then(setRuns).catch(() => setRuns([]));
  }, [open, refreshKey]);
  if (!open) return null;
  const trend = [...runs].reverse().map((run, index) => ({ n: index + 1, runtime: Number(run.runtime_seconds.toFixed(1)), cost: run.cost_usd }));
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/50" onClick={onClose}>
      <aside className="h-full w-full max-w-xl overflow-y-auto border-l border-slate-800 bg-canvas p-4" onClick={(event) => event.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-base font-semibold text-slate-100"><History size={17} /> Run history</h2>
          <button type="button" className="icon-button" aria-label="Close" onClick={onClose}><X size={14} /></button>
        </div>
        {trend.length > 1 && (
          <div className="mb-4 h-40 rounded-lg border border-slate-800 p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={trend}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="n" {...AXIS_PROPS} />
                <YAxis {...AXIS_PROPS} unit="s" width={40} />
                <Tooltip {...TOOLTIP_PROPS} />
                <Bar dataKey="runtime" name="Runtime (s)" fill={COLORS.cyan} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {runs.length === 0 && <p className="text-sm text-slate-500">No runs saved yet.</p>}
        <ul className="space-y-2">
          {runs.map((run) => (
            <li key={run.run_id}>
              <button type="button" onClick={() => { onSelect(run.run_id); onClose(); }}
                className="w-full rounded-lg border border-slate-800 bg-panel/80 p-3 text-left transition hover:border-cyan-500/40">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium text-slate-200">{run.dataset_name ?? "dataset"} → {run.target_column}</span>
                  <span className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] ${run.status === "SUCCESS" ? "bg-emerald-500/10 text-emerald-300" : "bg-rose-500/10 text-rose-300"}`}>{run.status}</span>
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {prettyName(run.model_name)} · {run.metric_name} {fmt(run.metric_value)} · {run.runtime_seconds.toFixed(1)}s · ${run.cost_usd.toFixed(5)} · {run.llm_calls} LLM calls{run.cache_hits ? ` · ${run.cache_hits} cached` : ""}
                </div>
                <div className="mt-0.5 text-[11px] text-slate-600">{new Date(run.created_at).toLocaleString()}</div>
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
