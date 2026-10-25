import { lazy, memo, Suspense } from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

const MiniChart = lazy(() => import("./MiniChart"));

interface MetricCardProps {
  value: string;
  label: string;
  trend: number | null;
  color: string;
  sparkline: number[];
  chart?: "area" | "bar" | "progress" | "none";
  progress?: number;
  footnote?: string;
}

function MetricCard({
  value,
  label,
  trend,
  color,
  sparkline,
  chart = "area",
  progress = 0,
  footnote,
}: MetricCardProps) {
  const TrendIcon = trend === null || trend === 0 ? Minus : trend > 0 ? ArrowUpRight : ArrowDownRight;
  const trendTone = trend === null || trend === 0 ? "text-slate-500" : trend > 0 ? "text-emerald-400" : "text-rose-400";

  return (
    <article className="metric-card group">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="eyebrow">{label}</p>
          <p className="mt-3 truncate text-[1.7rem] font-semibold tracking-[-0.045em] text-slate-50">{value}</p>
        </div>
        {chart !== "none" && chart !== "progress" && (
          <div className="h-11 w-[72px] opacity-90 transition-opacity group-hover:opacity-100">
            <Suspense fallback={<div className="h-full animate-pulse rounded bg-slate-800" />}>
              <MiniChart data={sparkline} color={color} variant={chart} />
            </Suspense>
          </div>
        )}
      </div>
      {chart === "progress" ? (
        <div className="mt-5">
          <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%`, backgroundColor: color }}
            />
          </div>
          <p className="mt-2 text-[10px] uppercase tracking-[.14em] text-slate-500">{progress.toFixed(0)}% of runs healthy</p>
        </div>
      ) : (
        <div className="mt-4 flex items-center justify-between gap-2 border-t border-slate-800/80 pt-3">
          <span className={`inline-flex items-center gap-1 text-[10px] font-medium ${trendTone}`}>
            <TrendIcon size={12} />
            {trend === null ? "live" : `${Math.abs(trend).toFixed(1)}%`}
          </span>
          <span className="truncate text-[10px] text-slate-500">{footnote ?? "vs previous run"}</span>
        </div>
      )}
    </article>
  );
}

export default memo(MetricCard);
