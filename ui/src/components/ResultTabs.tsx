import { memo, useCallback, useState, type ReactNode } from "react";
import {
  Braces,
  Check,
  ChevronDown,
  ChevronRight,
  Clipboard,
  ExternalLink,
  GitBranch,
  Layers3,
} from "lucide-react";
import type { ContractDefinition, FinalResult, TraceNode } from "../types";

const CONTRACTS: ContractDefinition[] = [
  {
    name: "PipelineState",
    description: "Mutable orchestration context",
    fields: {
      run_id: "UUID",
      dataset_uri: "ArtifactUri",
      task_type: "classification | regression",
      target_column: "string",
      prediction_goal: "string",
      status: "PipelineStatus",
      current_stage: "PipelineStage",
      retry_count: "integer ≥ 0",
      trace_id: "hex string",
    },
  },
  {
    name: "DataPrepResult",
    description: "Profiled, cleaned train/test artifacts",
    fields: {
      run_id: "UUID",
      dataset_hash: "SHA-256",
      schema_summary: "Record<string, string>",
      quality_findings: "string[]",
      target_column: "string",
      feature_columns: "string[]",
      row_count: "integer > 0",
      train_artifact_uri: "ArtifactUri",
      test_artifact_uri: "ArtifactUri",
    },
  },
  {
    name: "ModelSelectionResult",
    description: "Validated ADK A2A selection response",
    fields: {
      run_id: "UUID",
      candidate_run_ids: "string[]",
      best_run_id: "string",
      best_model_uri: "ArtifactUri",
      best_parameters: "Record<string, scalar>",
      training_metrics: "Record<string, number>",
    },
  },
  {
    name: "EvaluationResult",
    description: "Held-out gate decision and feedback",
    fields: {
      run_id: "UUID",
      model_uri: "ArtifactUri",
      held_out_metrics: "Record<string, number>",
      primary_metric: "string",
      acceptance_threshold: "number",
      metric_direction: "maximize | minimize",
      passed: "boolean",
      retry_feedback: "string | null",
    },
  },
  {
    name: "DeploymentResult",
    description: "Phase 3 deployment identity",
    fields: {
      run_id: "UUID",
      image_digest: "sha256 digest",
      service_url: "HttpUrl",
      health_status: "healthy | unhealthy",
      deployment_revision: "string",
      deployed_at: "datetime",
    },
  },
  {
    name: "PipelineResult",
    description: "Terminal machine-readable report card",
    fields: {
      run_id: "UUID",
      status: "completed | evaluation_completed | failed",
      model_card_markdown: "string",
      lineage: "Record<string, string>",
      trace_url: "HttpUrl",
      runtime_seconds: "number ≥ 0",
      input_tokens: "integer ≥ 0",
      output_tokens: "integer ≥ 0",
      error: "string | null",
    },
  },
];

interface ResultTabsProps {
  result: FinalResult | null;
  trace: TraceNode | null;
}

function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }, [value]);
  return (
    <button type="button" onClick={copy} className="icon-button" aria-label={label} title={label}>
      {copied ? <Check size={12} className="text-emerald-400" /> : <Clipboard size={12} />}
    </button>
  );
}

function ContractCard({ contract }: { contract: ContractDefinition }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="contract-card">
      <div className="flex items-center gap-2">
        <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => setOpen((value) => !value)}>
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <span className="truncate font-mono text-[11px] font-semibold text-slate-200">{contract.name}</span>
        </button>
        <CopyButton value={JSON.stringify(contract.fields, null, 2)} label={`Copy ${contract.name} JSON`} />
      </div>
      <p className="ml-5 mt-1 text-[10px] text-slate-500">{contract.description}</p>
      {open && (
        <dl className="mt-3 space-y-1.5 border-t border-slate-800 pt-3">
          {Object.entries(contract.fields).map(([name, type]) => (
            <div key={name} className="flex items-start justify-between gap-4 text-[10px]">
              <dt className="font-mono text-cyan-300/90">{name}</dt>
              <dd className="text-right text-slate-500">{type}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function TraceTreeNode({ node, depth = 0 }: { node: TraceNode; depth?: number }) {
  const [open, setOpen] = useState(depth < 1);
  const hasDetails = node.input != null || node.output != null || Boolean(node.children?.length);
  return (
    <div className={depth ? "ml-4 border-l border-slate-800 pl-3" : ""}>
      <button
        type="button"
        onClick={() => hasDetails && setOpen((value) => !value)}
        className="trace-node"
      >
        {hasDetails ? (open ? <ChevronDown size={12} /> : <ChevronRight size={12} />) : <span className="w-3" />}
        <span className={`trace-status ${node.status === "error" ? "bg-rose-400" : "bg-emerald-400"}`} />
        <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-300">{node.label ?? node.name}</span>
        {node.duration_ms != null && <span className="font-mono text-[9px] text-slate-600">{node.duration_ms}ms</span>}
      </button>
      {open && (
        <div className="pb-1">
          {(node.input != null || node.output != null) && (
            <div className="mb-2 ml-5 grid gap-2 xl:grid-cols-2">
              <pre className="trace-payload"><span>INPUT</span>{JSON.stringify(node.input, null, 2) ?? "—"}</pre>
              <pre className="trace-payload"><span>OUTPUT</span>{JSON.stringify(node.output, null, 2) ?? "—"}</pre>
            </div>
          )}
          {node.children?.map((child, index) => <TraceTreeNode key={`${child.name}-${index}`} node={child} depth={depth + 1} />)}
        </div>
      )}
    </div>
  );
}

function PipelineResultPanel({ result }: { result: FinalResult | null }) {
  if (!result) {
    return <EmptyState icon={<Layers3 size={18} />} text="Run the crew to generate a model report card." />;
  }
  const success = result.status === "SUCCESS";
  return (
    <div className="space-y-4">
      <div className={`result-banner ${success ? "result-success" : "result-failed"}`}>
        <span className="text-[9px] font-semibold uppercase tracking-[.18em] opacity-70">Terminal state</span>
        <strong className="mt-1 block text-sm tracking-tight">{result.status}</strong>
      </div>
      <div className="detail-grid">
        <div className="detail-row col-span-2">
          <span>Run ID</span>
          <div className="flex min-w-0 items-center gap-1">
            <code className="truncate">{result.run_id}</code>
            <CopyButton value={result.run_id} label="Copy run ID" />
          </div>
        </div>
        <div className="detail-row"><span>Model</span><strong>{result.model_name ?? "—"}</strong></div>
        <div className="detail-row"><span>Retries</span><strong>{result.retry_count}/2</strong></div>
        <div className="detail-row"><span>{result.metric_name ?? "Metric"}</span><strong>{result.metric_value?.toFixed(4) ?? "—"}</strong></div>
        <div className="detail-row"><span>Threshold</span><strong>{result.threshold?.toFixed(4) ?? "—"}</strong></div>
      </div>
      {(result.feedback || result.error) && (
        <div className="rounded-md border border-slate-800 bg-slate-950/50 p-3 text-[11px] leading-relaxed text-slate-400">
          {result.error ?? result.feedback}
        </div>
      )}
      {result.usage_report?.by_agent && Object.keys(result.usage_report.by_agent).length > 0 && (
        <div className="rounded-md border border-slate-800 bg-slate-950/50 p-3 text-[11px] text-slate-400">
          <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">LLM usage &amp; cost by agent</h4>
          <table className="w-full text-left">
            <thead>
              <tr className="text-slate-500"><th>Agent</th><th>Calls</th><th>Tokens</th><th>Latency</th><th>Cost</th></tr>
            </thead>
            <tbody>
              {Object.entries(result.usage_report.by_agent).map(([agent, bucket]) => (
                <tr key={agent}>
                  <td className="pr-2">{agent}</td>
                  <td>{bucket.requests}</td>
                  <td>{bucket.total_tokens.toLocaleString()}</td>
                  <td>{bucket.latency_seconds.toFixed(1)}s</td>
                  <td>${bucket.estimated_cost_usd.toFixed(5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.llm_summary && (
        <div className="llm-insight">
          <h4>LLM run summary</h4>
          <p>{result.llm_summary}</p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        <a className="action-link action-primary" href={result.trace_url} target="_blank" rel="noreferrer">
          View Langfuse <ExternalLink size={12} />
        </a>
        <a className="action-link" href={result.mlflow_url} target="_blank" rel="noreferrer">
          View MLflow <ExternalLink size={12} />
        </a>
      </div>
    </div>
  );
}

function EmptyState({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <div className="flex min-h-60 flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/20 text-center">
      <div className="mb-3 rounded-lg border border-slate-800 bg-slate-900 p-2 text-slate-500">{icon}</div>
      <p className="max-w-52 text-[11px] leading-relaxed text-slate-500">{text}</p>
    </div>
  );
}

function ResultTabs({ result, trace }: ResultTabsProps) {
  const [tab, setTab] = useState<"pipeline" | "contracts" | "traces">("pipeline");
  const tabs = [
    { id: "pipeline" as const, label: "Pipeline", icon: <Layers3 size={12} /> },
    { id: "contracts" as const, label: "Contracts", icon: <Braces size={12} /> },
    { id: "traces" as const, label: "Traces", icon: <GitBranch size={12} /> },
  ];
  return (
    <section className="panel min-h-[620px]">
      <div className="flex border-b border-slate-800 px-4 pt-3">
        {tabs.map((item) => (
          <button
            type="button"
            key={item.id}
            onClick={() => setTab(item.id)}
            className={`tab-button ${tab === item.id ? "tab-active" : ""}`}
          >
            {item.icon}{item.label}
          </button>
        ))}
      </div>
      <div className="p-4">
        {tab === "pipeline" && <PipelineResultPanel result={result} />}
        {tab === "contracts" && <div className="space-y-2">{CONTRACTS.map((contract) => <ContractCard key={contract.name} contract={contract} />)}</div>}
        {tab === "traces" && (
          trace ? (
            <div>
              <div className="mb-3 flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-[.14em] text-slate-500">Langfuse span tree</span>
                {result && <a className="action-link !w-auto px-2.5 py-1.5" href={result.trace_url} target="_blank" rel="noreferrer">Open trace <ExternalLink size={11} /></a>}
              </div>
              <TraceTreeNode node={trace} />
            </div>
          ) : <EmptyState icon={<GitBranch size={18} />} text="Trace inputs and outputs appear here after the first run." />
        )}
      </div>
    </section>
  );
}

export default memo(ResultTabs);
