import { ArrowLeftRight, BadgeCheck, Bot, Cpu, FileJson } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { A2ALog, JsonValue, ResultPayload, UsageCall } from "../../types";
import { AXIS_PROPS, COLORS, Card, EmptyHint, STAGE_LABELS, SourceBadge, Stat, TOOLTIP_PROPS, fmt } from "./shared";

const FRAMEWORK_STYLE: Record<string, string> = {
  crewai: "border-orange-500/30 bg-orange-500/10 text-orange-300",
  adk: "border-sky-500/30 bg-sky-500/10 text-sky-300",
};

function DecisionRow({ call }: { call: UsageCall }) {
  return (
    <tr className="border-b border-slate-800/60 align-top">
      <td className="py-2.5 pr-3">
        <div className="text-sm font-medium text-slate-200">{STAGE_LABELS[call.stage] ?? call.stage}</div>
        <div className="mt-0.5 flex items-center gap-1.5 text-xs text-slate-500">
          <span className={`rounded border px-1.5 py-px text-[10px] font-semibold uppercase ${FRAMEWORK_STYLE[call.framework] ?? "border-slate-700 text-slate-400"}`}>{call.framework}</span>
          {call.agent}
        </div>
      </td>
      <td className="py-2.5 pr-3"><SourceBadge source={call.source} provider={call.provider} /></td>
      <td className="py-2.5 pr-3 font-mono text-xs text-slate-400">{call.model || "—"}</td>
      <td className="py-2.5 pr-3 text-right font-mono text-xs text-slate-300">{call.total_tokens ? call.total_tokens.toLocaleString() : "—"}</td>
      <td className="py-2.5 pr-3 text-right font-mono text-xs text-slate-300">{call.latency_seconds ? `${call.latency_seconds.toFixed(1)}s` : "—"}</td>
      <td className="py-2.5 text-right font-mono text-xs text-slate-300">${(call.estimated_cost_usd ?? 0).toFixed(5)}</td>
    </tr>
  );
}

function cardValue(card: Record<string, JsonValue>, key: string): string {
  const value = card[key];
  return typeof value === "string" ? value : value == null ? "—" : JSON.stringify(value);
}

function A2APanel({ a2a }: { a2a?: A2ALog | null }) {
  if (!a2a || (!a2a.exchanges?.length && !Object.keys(a2a.agent_card ?? {}).length)) {
    return <EmptyHint>The A2A exchange appears once the ADK agent has been called.</EmptyHint>;
  }
  const card = a2a.agent_card ?? {};
  const interfaces = Array.isArray(card.supportedInterfaces) ? (card.supportedInterfaces as Record<string, JsonValue>[]) : [];
  const skills = Array.isArray(card.skills) ? (card.skills as Record<string, JsonValue>[]) : [];
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="rounded-lg border border-sky-500/20 bg-sky-500/[.04] p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-sky-200"><BadgeCheck size={16} /> Agent Card (discovered)</div>
        <dl className="mt-3 space-y-2 text-xs">
          <div><dt className="text-slate-500">Name</dt><dd className="font-mono text-slate-200">{cardValue(card, "name")}</dd></div>
          <div><dt className="text-slate-500">Description</dt><dd className="text-slate-300">{cardValue(card, "description")}</dd></div>
          <div><dt className="text-slate-500">Card URL</dt><dd className="break-all font-mono text-slate-300">{cardValue(card, "card_url")}</dd></div>
          {interfaces.map((item, index) => (
            <div key={index}><dt className="text-slate-500">Interface</dt><dd className="font-mono text-slate-300">{String(item.protocolBinding ?? "JSONRPC")} · {String(item.url ?? "")} · v{String(item.protocolVersion ?? "")}</dd></div>
          ))}
          {skills.length > 0 && (
            <div><dt className="text-slate-500">Skills</dt><dd className="text-slate-300">{skills.map((skill) => String(skill.name ?? skill.id)).join(", ")}</dd></div>
          )}
        </dl>
      </div>
      <div className="space-y-2">
        {a2a.exchanges.map((exchange) => {
          const request = exchange.request as Record<string, JsonValue>;
          const prep = (request.data_prep_result ?? {}) as Record<string, JsonValue>;
          return (
            <div key={exchange.attempt} className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="flex items-center gap-2 text-sm font-medium text-slate-200"><ArrowLeftRight size={15} className="text-sky-300" /> Attempt {exchange.attempt} · {exchange.method}</span>
                <span className="font-mono text-xs text-slate-500">{exchange.task_state} · {exchange.latency_ms} ms</span>
              </div>
              <div className="mt-2 grid gap-2 md:grid-cols-2">
                <div className="rounded border border-slate-800 p-2 text-xs text-slate-400">
                  <div className="mb-1 font-semibold text-orange-300">CrewAI → ADK (request)</div>
                  task: {String(request.task_type)} · features: {String(prep.feature_count)} · train rows: {String(prep.train_rows)}
                  {request.retry_feedback ? <div className="mt-1 text-amber-200/80">feedback: {String(request.retry_feedback)}</div> : <div className="mt-1 text-slate-500">first attempt — no feedback</div>}
                </div>
                <div className="rounded border border-slate-800 p-2 text-xs text-slate-400">
                  <div className="mb-1 font-semibold text-sky-300">ADK → CrewAI (response)</div>
                  best: {String(exchange.response.best_model)} · score {fmt(Number(exchange.response.metric_value))}
                  <div className="mt-1 break-all font-mono text-[10px] text-slate-500">task {exchange.task_id}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function AgentsTab({ payload }: { payload: ResultPayload }) {
  const usage = payload.final_result.usage_report ?? {};
  const calls = usage.calls ?? [];
  const byAgent = Object.entries(usage.by_agent ?? {}).map(([agent, bucket]) => ({ agent, tokens: bucket.total_tokens, calls: bucket.requests, cost: bucket.estimated_cost_usd }));
  const narratives = payload.llm_insights;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="LLM calls" value={usage.llm_requests ?? 0} tone="cyan" />
        <Stat label="Local (Ollama)" value={usage.local_llm_requests ?? 0} tone="emerald" hint="$0 per call" />
        <Stat label="Cache hits" value={usage.cache_hits ?? 0} tone="amber" hint="reused decisions" />
        <Stat label="Default rules used" value={usage.fallback_decisions ?? 0} tone={(usage.fallback_decisions ?? 0) ? "rose" : "slate"} />
        <Stat label="Tokens" value={(usage.total_tokens ?? 0).toLocaleString()} hint={`${usage.input_tokens ?? 0} in / ${usage.output_tokens ?? 0} out`} />
        <Stat label="LLM cost" value={`$${(usage.estimated_cost_usd ?? 0).toFixed(5)}`} tone="violet" hint={`${(usage.llm_latency_seconds ?? 0).toFixed(1)}s in LLMs`} />
      </div>

      <Card title="Who decided what" subtitle="Every agent decision, where it came from, and what it cost" action={<Cpu size={16} className="text-cyan-300" />}>
        {calls.length ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left">
              <thead>
                <tr className="border-b border-slate-800 text-xs text-slate-500">
                  <th className="pb-2 font-medium">Decision</th>
                  <th className="pb-2 font-medium">Source</th>
                  <th className="pb-2 font-medium">Model</th>
                  <th className="pb-2 text-right font-medium">Tokens</th>
                  <th className="pb-2 text-right font-medium">Latency</th>
                  <th className="pb-2 text-right font-medium">Cost</th>
                </tr>
              </thead>
              <tbody>{calls.map((call, index) => <DecisionRow key={`${call.stage}-${index}`} call={call} />)}</tbody>
            </table>
          </div>
        ) : (
          <EmptyHint>No LLM decisions were recorded (LLM mode offline).</EmptyHint>
        )}
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Tokens by agent" subtitle="Where the LLM budget went">
          {byAgent.length ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={byAgent} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                  <XAxis type="number" {...AXIS_PROPS} />
                  <YAxis type="category" dataKey="agent" width={190} {...AXIS_PROPS} />
                  <Tooltip {...TOOLTIP_PROPS} />
                  <Bar dataKey="tokens" fill={COLORS.violet} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyHint>No token usage recorded.</EmptyHint>}
        </Card>
        <Card title="Agent reasoning" subtitle="What each agent said about its decision" action={<Bot size={16} className="text-violet-300" />}>
          <div className="space-y-3 text-xs leading-relaxed text-slate-300">
            {narratives?.data_prep && <div><div className="mb-1 font-semibold text-orange-300">Data Preparation Engineer (CrewAI)</div><p className="whitespace-pre-line">{narratives.data_prep}</p></div>}
            {narratives?.model_selection && <div><div className="mb-1 font-semibold text-sky-300">Model Selection Agent (ADK)</div><p className="whitespace-pre-line">{narratives.model_selection}</p></div>}
            {narratives?.evaluation && <div><div className="mb-1 font-semibold text-orange-300">Model Evaluation Engineer (CrewAI)</div><p className="whitespace-pre-line">{narratives.evaluation}</p></div>}
            {!narratives && <EmptyHint>No reasoning recorded.</EmptyHint>}
          </div>
        </Card>
      </div>

      <Card title="Agent-to-Agent (A2A) protocol" subtitle="CrewAI orchestrator ↔ Google ADK model-selection agent" action={<FileJson size={16} className="text-sky-300" />}>
        <A2APanel a2a={payload.a2a} />
      </Card>
    </div>
  );
}
