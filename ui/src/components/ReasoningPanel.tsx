import { BrainCircuit, ChevronDown } from "lucide-react";
import type { PipelineStage } from "../types";
import ReasoningDisplay from "./ReasoningDisplay";

interface ReasoningPanelProps {
  stage: PipelineStage;
  defaultOpen?: boolean;
}

export default function ReasoningPanel({ stage, defaultOpen = false }: ReasoningPanelProps) {
  if (!stage.reasoning || Object.keys(stage.reasoning).length === 0) return null;
  return (
    <details className={`reasoning-panel reasoning-${stage.status}`} open={defaultOpen}>
      <summary>
        <span className="reasoning-icon"><BrainCircuit size={14} /></span>
        <span className="min-w-0 flex-1">
          <strong>{stage.name}</strong>
          <small>{stage.detail}</small>
        </span>
        <span className="reasoning-status">{stage.status}</span>
        <ChevronDown className="reasoning-chevron" size={14} />
      </summary>
      <ReasoningDisplay reasoning={stage.reasoning} />
    </details>
  );
}
