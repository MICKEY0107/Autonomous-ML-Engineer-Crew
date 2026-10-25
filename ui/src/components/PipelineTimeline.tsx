import { memo } from "react";
import { Activity, RotateCcw } from "lucide-react";
import type { PipelineStage } from "../types";
import TimelineStage from "./TimelineStage";

interface PipelineTimelineProps {
  stages: PipelineStage[];
  retryCount: number;
}

function PipelineTimeline({ stages, retryCount }: PipelineTimelineProps) {
  const completeCount = stages.filter((stage) => stage.status === "success" || stage.status === "failed").length;
  return (
    <section className="panel min-h-[620px]">
      <header className="panel-header">
        <div>
          <div className="flex items-center gap-2">
            <Activity size={14} className="text-cyan-400" />
            <h2 className="panel-title">Pipeline Progress</h2>
          </div>
          <p className="panel-subtitle">CrewAI orchestration · live stage telemetry</p>
        </div>
        <div className="flex items-center gap-2">
          {retryCount > 0 && (
            <span className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/20 bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-300">
              <RotateCcw size={11} /> {retryCount}/2 retries
            </span>
          )}
          <span className="rounded-md border border-slate-700 bg-slate-900/70 px-2 py-1 font-mono text-[10px] text-slate-400">
            {completeCount}/{stages.length}
          </span>
        </div>
      </header>
      <ol className="px-5 py-5">
        {stages.map((stage, index) => (
          <TimelineStage key={stage.name} stage={stage} isLast={index === stages.length - 1} />
        ))}
      </ol>
    </section>
  );
}

export default memo(PipelineTimeline);
