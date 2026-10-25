import { memo } from "react";
import { motion } from "framer-motion";
import { Check, Clock3, RefreshCw, X } from "lucide-react";
import type { PipelineStage } from "../types";

interface TimelineStageProps {
  stage: PipelineStage;
  isLast: boolean;
}

function formatDuration(duration: number): string {
  if (!duration) return "—";
  return duration < 1000 ? `${duration}ms` : `${(duration / 1000).toFixed(2)}s`;
}

function TimelineStageView({ stage, isLast }: TimelineStageProps) {
  const icon = stage.status === "success"
    ? <Check size={12} />
    : stage.status === "failed"
      ? <X size={12} />
      : stage.status === "retry"
        ? <RefreshCw size={12} />
        : <Clock3 size={11} />;

  return (
    <motion.li
      layout
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.28 }}
      className={`timeline-row ${stage.status === "running" ? "timeline-row-running" : ""}`}
    >
      <div className="relative flex w-8 shrink-0 justify-center">
        <span className={`stage-dot stage-dot-${stage.status}`}>{icon}</span>
        {!isLast && <span className={`stage-line ${stage.status === "success" ? "stage-line-complete" : ""}`} />}
      </div>
      <div className="min-w-0 flex-1 pb-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-medium text-slate-100">{stage.name}</h3>
              {stage.attempts > 1 && (
                <span className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-amber-300">
                  {stage.attempts} attempts
                </span>
              )}
            </div>
            <p className="mt-1 truncate text-[11px] text-slate-500">{stage.detail}</p>
          </div>
          <motion.span
            key={`${stage.status}-${stage.duration_ms}`}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            className={`shrink-0 font-mono text-[11px] ${stage.status === "running" ? "text-cyan-300" : "text-slate-400"}`}
          >
            {stage.status === "retry" ? `↻ retry ${Math.max(1, stage.attempts)}/2` : formatDuration(stage.duration_ms)}
          </motion.span>
        </div>
      </div>
    </motion.li>
  );
}

export default memo(TimelineStageView);
