import { memo, useCallback, useMemo, useState, type ChangeEvent } from "react";
import { ChevronDown, FileSpreadsheet, Play, RotateCcw, UploadCloud } from "lucide-react";
import { motion } from "framer-motion";
import type { RunInput } from "../types";

interface InputPanelProps {
  isRunning: boolean;
  statusLabel: string;
  onRun: (input: RunInput) => Promise<void>;
}

function parseCsvHeader(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else quoted = !quoted;
    } else if (char === "," && !quoted) {
      fields.push(current.trim());
      current = "";
    } else current += char;
  }
  fields.push(current.trim());
  return fields.filter(Boolean);
}

function InputPanel({ isRunning, statusLabel, onRun }: InputPanelProps) {
  const [expanded, setExpanded] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [goal, setGoal] = useState("predict house price accurately");
  const [forceFail, setForceFail] = useState(false);
  const [dragging, setDragging] = useState(false);

  const loadFile = useCallback(async (selected: File | null) => {
    if (!selected) return;
    setFile(selected);
    const header = (await selected.slice(0, 16_384).text()).split(/\r?\n/, 1)[0] ?? "";
    const nextColumns = parseCsvHeader(header);
    setColumns(nextColumns);
    setTarget((existing) => existing && nextColumns.includes(existing) ? existing : nextColumns.at(-1) ?? "");
  }, []);

  const canRun = Boolean(file && target.trim() && goal.trim() && !isRunning);
  const fileLabel = useMemo(() => file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : "Drop a CSV here or browse", [file]);

  const submit = useCallback(async () => {
    if (!file || !canRun) return;
    await onRun({ file, targetColumn: target, predictionGoal: goal, forceFailEval: forceFail });
  }, [canRun, file, forceFail, goal, onRun, target]);

  const handleFileChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    // Resetting permits selecting the same CSV again after a correction.
    event.target.value = "";
    void loadFile(selected);
  }, [loadFile]);

  return (
    <section className="panel overflow-hidden">
      <button type="button" onClick={() => setExpanded((value) => !value)} className="panel-header w-full text-left">
        <div>
          <div className="flex items-center gap-2"><UploadCloud size={14} className="text-emerald-400" /><h2 className="panel-title">Input & Controls</h2></div>
          <p className="panel-subtitle">Validated CSV upload · guarded prediction goal · bounded retry demo</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`status-chip ${isRunning ? "status-running" : statusLabel === "Complete" ? "status-success" : "status-idle"}`}>{statusLabel}</span>
          <ChevronDown size={15} className={`text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`} />
        </div>
      </button>
      {expanded && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="grid gap-4 border-t border-slate-800 p-4 lg:grid-cols-[1.35fr_1fr_1.35fr_auto]">
          <label
            htmlFor="csv-upload"
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); void loadFile(event.dataTransfer.files[0] ?? null); }}
            className={`upload-zone ${dragging ? "upload-zone-active" : ""}`}
          >
            <input id="csv-upload" type="file" accept=".csv,text/csv" className="sr-only" onChange={handleFileChange} />
            <FileSpreadsheet size={18} className={file ? "text-emerald-400" : "text-slate-500"} />
            <div className="min-w-0"><p className="truncate text-[11px] font-medium text-slate-300">{fileLabel}</p><p className="mt-0.5 text-[9px] uppercase tracking-wider text-slate-600">CSV · max 100 MB</p></div>
          </label>
          <label className="field-label">
            Target column
            {columns.length ? (
              <select value={target} onChange={(event) => setTarget(event.target.value)} className="field-input">
                {columns.map((column) => <option key={column} value={column}>{column}</option>)}
              </select>
            ) : <input value={target} onChange={(event) => setTarget(event.target.value)} placeholder="sale_price" className="field-input" />}
          </label>
          <label className="field-label">
            Prediction goal
            <input value={goal} onChange={(event) => setGoal(event.target.value)} className="field-input" />
          </label>
          <div className="flex min-w-[190px] flex-col justify-between gap-3">
            <label className="flex cursor-pointer items-center justify-between gap-3 text-[10px] text-slate-400">
              <span className="inline-flex items-center gap-1.5"><RotateCcw size={11} /> Retry Loop Demo</span>
              <input type="checkbox" checked={forceFail} onChange={(event) => setForceFail(event.target.checked)} className="demo-toggle" />
            </label>
            <button type="button" onClick={() => void submit()} disabled={!canRun} className="run-button">
              {isRunning ? <span className="run-spinner" /> : <Play size={13} fill="currentColor" />}
              {isRunning ? "Crew running" : "Run pipeline"}
            </button>
          </div>
        </motion.div>
      )}
    </section>
  );
}

export default memo(InputPanel);
