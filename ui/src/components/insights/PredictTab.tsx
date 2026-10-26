import { useEffect, useMemo, useState } from "react";
import { Download, Rocket, Server, Sparkles, Upload } from "lucide-react";
import { apiErrorMessage, getDeployment, predictCsv, predictRows } from "../../api";
import type { DeploymentMetadata, FinalResult, Prediction } from "../../types";
import { Card, EmptyHint, Stat, fmt, prettyName } from "./shared";

export default function PredictTab({ final }: { final: FinalResult }) {
  const runHex = final.run_id.replace(/-/g, "");
  const [meta, setMeta] = useState<DeploymentMetadata | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [apiKey, setApiKey] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setMeta(null);
    setResult(null);
    if (!final.deployment) return undefined;
    getDeployment(runHex)
      .then((data) => {
        if (cancelled) return;
        setMeta(data);
        setValues(Object.fromEntries(data.features.map((feature) => [feature.name, String(feature.example ?? "")])));
      })
      .catch((requestError) => !cancelled && setError(apiErrorMessage(requestError)));
    return () => { cancelled = true; };
  }, [final.deployment, runHex]);

  const row = useMemo(() => {
    if (!meta) return {};
    return Object.fromEntries(meta.features.map((feature) => {
      const raw = values[feature.name] ?? "";
      if (raw === "") return [feature.name, null];
      return [feature.name, feature.type === "number" ? Number(raw) : raw];
    }));
  }, [meta, values]);

  if (!final.deployment) {
    return (
      <EmptyHint>
        {final.deployment_error
          ? `Deployment was skipped: ${final.deployment_error}`
          : "Only a model that passes the held-out quality check is deployed. Run a dataset that passes to get a live prediction endpoint."}
      </EmptyHint>
    );
  }

  const predict = async () => {
    setBusy(true);
    setError(null);
    try {
      const [prediction] = await predictRows(runHex, [row], apiKey || undefined);
      setResult(prediction);
    } catch (requestError) {
      setError(apiErrorMessage(requestError));
    } finally {
      setBusy(false);
    }
  };

  const batch = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const blob = await predictCsv(runHex, file, apiKey || undefined);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `predictions-${runHex.slice(0, 8)}.csv`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(apiErrorMessage(requestError));
    } finally {
      setBusy(false);
    }
  };

  const deployment = final.deployment;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label="Deployment" value={deployment.health_status === "healthy" ? "Healthy" : "Unhealthy"} tone={deployment.health_status === "healthy" ? "emerald" : "rose"} />
        <Stat label="Revision" value={deployment.deployment_revision} />
        <Stat label="Model" value={prettyName(meta?.model_name ?? final.model_name)} tone="cyan" />
        <Stat label="Digest" value={deployment.image_digest.slice(7, 19)} hint="SHA-256 of the deployed bundle" />
      </div>
      <Card title="Prediction endpoint" subtitle="Use it from any app with an HTTP POST" action={<Server size={16} className="text-emerald-300" />}>
        <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-slate-300">{`POST ${deployment.service_url}/predict
Content-Type: application/json${meta?.auth_required ? "\nX-API-Key: <your key>" : ""}

{"rows": [${JSON.stringify(row)}]}`}</pre>
      </Card>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Card title="Try a prediction" subtitle="Pre-filled with typical values from the training data" action={<Sparkles size={16} className="text-amber-300" />}>
          {!meta ? <EmptyHint>Loading the model's input schema…</EmptyHint> : (
            <>
              <div className="grid gap-3 sm:grid-cols-2">
                {meta.features.map((feature) => (
                  <label key={feature.name} className="flex flex-col gap-1 text-xs text-slate-400">
                    {feature.name}
                    {feature.type === "category" ? (
                      <select className="field-input text-sm" value={values[feature.name] ?? ""} onChange={(event) => setValues({ ...values, [feature.name]: event.target.value })}>
                        <option value="">(unknown)</option>
                        {(feature.categories ?? []).map((category) => <option key={category} value={category}>{category}</option>)}
                      </select>
                    ) : (
                      <input
                        className="field-input text-sm"
                        type="number"
                        value={values[feature.name] ?? ""}
                        placeholder={feature.min != null ? `${fmt(feature.min)} – ${fmt(feature.max)}` : ""}
                        onChange={(event) => setValues({ ...values, [feature.name]: event.target.value })}
                      />
                    )}
                  </label>
                ))}
              </div>
              {meta.auth_required && (
                <label className="mt-3 flex flex-col gap-1 text-xs text-slate-400">
                  API key
                  <input className="field-input text-sm" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} />
                </label>
              )}
              <button type="button" className="run-button mt-4" disabled={busy} onClick={() => void predict()}>
                <Rocket size={14} /> Predict {meta.target_column}
              </button>
            </>
          )}
        </Card>
        <div className="space-y-4">
          <Card title="Result">
            {result ? (
              <div>
                <div className="text-xs uppercase tracking-wide text-slate-500">Predicted {meta?.target_column}</div>
                <div className="mt-1 text-3xl font-semibold text-emerald-300">{typeof result.prediction === "number" ? fmt(result.prediction) : result.prediction}</div>
                {result.probabilities && (
                  <ul className="mt-3 space-y-1.5">
                    {Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([label, probability]) => (
                      <li key={label} className="text-xs text-slate-300">
                        <div className="flex justify-between"><span>{label}</span><span className="font-mono">{(probability * 100).toFixed(1)}%</span></div>
                        <div className="mt-1 h-1.5 rounded-full bg-slate-800"><div className="h-1.5 rounded-full bg-cyan-400" style={{ width: `${probability * 100}%` }} /></div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ) : <p className="text-sm text-slate-500">Fill in the inputs and press Predict.</p>}
          </Card>
          <Card title="Score a whole file" subtitle="Upload a CSV with the same columns; get it back with predictions">
            <label className="upload-zone">
              <Upload size={16} className="text-cyan-300" />
              <span className="text-sm text-slate-300">Choose CSV…</span>
              <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(event) => void batch(event.target.files?.[0])} />
              <Download size={14} className="ml-auto text-slate-500" />
            </label>
          </Card>
          {error && <p className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">{error}</p>}
        </div>
      </div>
    </div>
  );
}
