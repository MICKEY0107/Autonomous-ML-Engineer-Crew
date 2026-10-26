import type { ReactNode } from "react";

export const COLORS = {
  cyan: "#22d3ee",
  emerald: "#34d399",
  violet: "#a78bfa",
  amber: "#fbbf24",
  rose: "#fb7185",
  slate: "#64748b",
  grid: "#1e293b",
  axis: "#94a3b8",
};

export const AXIS_PROPS = {
  stroke: COLORS.grid,
  tick: { fill: COLORS.axis, fontSize: 11 },
  tickLine: false,
} as const;

export const TOOLTIP_PROPS = {
  contentStyle: {
    background: "#0b1016",
    border: "1px solid #334155",
    borderRadius: 8,
    fontSize: 12,
    color: "#e2e8f0",
  },
  labelStyle: { color: "#94a3b8" },
  cursor: { fill: "rgba(148,163,184,0.08)" },
} as const;

export function prettyName(value: string | null | undefined): string {
  if (!value) return "—";
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function fmt(value: number | null | undefined, digits = 3): string {
  if (value == null || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(value);
  if (abs >= 1_000) return Math.round(value).toLocaleString();
  if (abs >= 100) return value.toFixed(1);
  return Number(value.toPrecision(digits)).toString();
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value == null) return "—";
  return `${value.toFixed(digits)}%`;
}

export function Card({ title, subtitle, action, children, className = "" }: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-panel/90 ${className}`}>
      <header className="flex items-start justify-between gap-3 border-b border-slate-800/70 px-4 py-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {action}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, tone = "slate" }: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "slate" | "emerald" | "rose" | "amber" | "cyan" | "violet";
}) {
  const tones: Record<string, string> = {
    slate: "text-slate-100",
    emerald: "text-emerald-300",
    rose: "text-rose-300",
    amber: "text-amber-300",
    cyan: "text-cyan-300",
    violet: "text-violet-300",
  };
  return (
    <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 truncate text-lg font-semibold ${tones[tone]}`}>{value}</div>
      {hint && <div className="mt-0.5 truncate text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

const SOURCE_STYLES: Record<string, { label: string; className: string }> = {
  llm: { label: "Primary LLM", className: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300" },
  secondary_llm: { label: "Fallback LLM", className: "border-violet-500/30 bg-violet-500/10 text-violet-300" },
  narrative_llm: { label: "Local LLM", className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" },
  cache: { label: "Cache", className: "border-amber-500/30 bg-amber-500/10 text-amber-300" },
  fallback: { label: "Default rules", className: "border-slate-600 bg-slate-800/60 text-slate-300" },
};

export function SourceBadge({ source, provider }: { source: string; provider?: string }) {
  const local = provider?.toLowerCase() === "ollama";
  const style = local && source !== "cache" && source !== "fallback"
    ? { label: "Local LLM (Ollama)", className: SOURCE_STYLES.narrative_llm.className }
    : SOURCE_STYLES[source] ?? SOURCE_STYLES.fallback;
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${style.className}`}>
      {style.label}
    </span>
  );
}

export function EmptyHint({ children }: { children: ReactNode }) {
  return (
    <div className="grid min-h-40 place-items-center rounded-lg border border-dashed border-slate-800 p-6 text-center text-sm text-slate-500">
      {children}
    </div>
  );
}

export const STAGE_LABELS: Record<string, string> = {
  data_prep_plan: "Data preparation plan",
  model_search_plan: "Model search plan",
  model_choice: "Model choice",
  evaluation_reflection: "Evaluation reflection",
  pipeline_summary: "Run summary",
  evaluation: "Evaluation narrative",
};
