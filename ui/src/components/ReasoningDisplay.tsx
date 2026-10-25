import { ArrowRight, Braces, ListChecks, RotateCcw, Sparkles } from "lucide-react";
import type { JsonValue, ReasoningData } from "../types";
import MetricComparison from "./MetricComparison";

interface ReasoningDisplayProps {
  reasoning: ReasoningData;
}

function isRecord(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function recordArray(value: JsonValue | undefined): Array<Record<string, JsonValue>> {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function humanize(key: string) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatValue(value: JsonValue): string {
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

const SPECIAL_KEYS = new Set([
  "candidates",
  "candidate_history",
  "metrics",
  "operations",
  "trade_offs",
  "retry_history",
  "selection_reason",
  "evaluation_reason",
  "decision",
  "llm_reasoning",
  "llm_summary",
]);

function TextList({ title, values }: { title: string; values: JsonValue[] }) {
  if (!values.length) return null;
  return (
    <div className="reasoning-block">
      <h4><ListChecks size={12} /> {humanize(title)}</h4>
      <ul>{values.map((value, index) => <li key={`${title}-${index}`}>{formatValue(value)}</li>)}</ul>
    </div>
  );
}

export default function ReasoningDisplay({ reasoning }: ReasoningDisplayProps) {
  const candidates = recordArray(reasoning.candidates);
  const candidateHistory = recordArray(reasoning.candidate_history);
  const metrics = recordArray(reasoning.metrics);
  const retryHistory = recordArray(reasoning.retry_history);
  const summary = reasoning.selection_reason ?? reasoning.evaluation_reason ?? reasoning.decision;
  const llmInsight = reasoning.llm_reasoning ?? reasoning.llm_summary;
  const regularEntries = Object.entries(reasoning).filter(([key]) => !SPECIAL_KEYS.has(key));

  return (
    <div className="reasoning-content">
      {summary != null && (
        <div className="reasoning-callout"><ArrowRight size={13} /><span>{formatValue(summary)}</span></div>
      )}

      {typeof llmInsight === "string" && llmInsight.length > 0 && (
        <div className="llm-insight">
          <h4><Sparkles size={12} /> Gemini reasoning</h4>
          <p>{llmInsight}</p>
        </div>
      )}

      {candidates.length > 0 && (
        <div className="reasoning-block">
          <h4><Braces size={12} /> Candidate comparison</h4>
          <MetricComparison rows={candidates} />
        </div>
      )}

      {metrics.length > 0 && (
        <div className="reasoning-block">
          <h4><Braces size={12} /> Held-out metrics</h4>
          <MetricComparison rows={metrics} />
        </div>
      )}

      {Array.isArray(reasoning.operations) && <TextList title="operations" values={reasoning.operations} />}
      {Array.isArray(reasoning.trade_offs) && <TextList title="trade_offs" values={reasoning.trade_offs} />}

      {retryHistory.length > 0 && (
        <div className="reasoning-block">
          <h4><RotateCcw size={12} /> Retry history</h4>
          <MetricComparison rows={retryHistory} />
        </div>
      )}

      {candidateHistory.length > candidates.length && (
        <details className="reasoning-history">
          <summary>All candidate attempts ({candidateHistory.length})</summary>
          <MetricComparison rows={candidateHistory} />
        </details>
      )}

      {regularEntries.length > 0 && (
        <dl className="reasoning-facts">
          {regularEntries.map(([key, value]) => (
            <div key={key}>
              <dt>{humanize(key)}</dt>
              <dd className={typeof value === "object" && value != null ? "reasoning-json" : ""}>
                {formatValue(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
