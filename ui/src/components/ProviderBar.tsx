import { useEffect, useState } from "react";
import { Database, RefreshCw } from "lucide-react";
import { getProviders } from "../api";
import type { ProvidersResponse } from "../types";

const NAMES: Record<string, string> = { gemini: "Gemini", ollama: "Ollama (local)", openrouter: "OpenRouter" };

export default function ProviderBar() {
  const [state, setState] = useState<ProvidersResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async (refresh = false) => {
    setLoading(true);
    try {
      setState(await getProviders(refresh));
    } catch {
      setState(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  if (!state) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {state.mode !== "live" && (
        <span className="rounded-full border border-slate-600 px-2 py-0.5 text-[11px] text-slate-400">LLM offline mode</span>
      )}
      {state.providers.map((provider) => (
        <span
          key={`${provider.role}-${provider.model}`}
          title={`${provider.role}: ${provider.model}\n${provider.detail}`}
          className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] ${provider.ok ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" : "border-rose-500/30 bg-rose-500/10 text-rose-300"}`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${provider.ok ? "bg-emerald-400" : "bg-rose-400"}`} />
          {NAMES[provider.provider] ?? provider.provider}
          <span className="hidden text-slate-500 lg:inline">· {provider.role}</span>
        </span>
      ))}
      <span title={`${state.cache.entries} cached decisions`} className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-300">
        <Database size={11} /> cache {state.cache.enabled ? `on · ${state.cache.entries}` : "off"}
      </span>
      <button type="button" aria-label="Re-check providers" className="icon-button" onClick={() => void load(true)}>
        <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
      </button>
    </div>
  );
}
