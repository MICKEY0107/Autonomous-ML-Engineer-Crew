import { AlertTriangle, Fingerprint } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DatasetInsights } from "../../types";
import { AXIS_PROPS, COLORS, Card, EmptyHint, Stat, TOOLTIP_PROPS, fmt, pct } from "./shared";

export default function DataTab({ data }: { data?: DatasetInsights }) {
  if (!data) return <EmptyHint>Dataset insights are not available for this run.</EmptyHint>;
  const missing = data.missing_by_column.filter((row) => row.missing > 0);
  const isClass = data.task_type === "classification";
  const distribution = data.target_distribution.map((row) => ({ name: row.label ?? row.bin ?? "", count: row.count }));
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Rows" value={data.rows.toLocaleString()} hint={data.train_rows ? `${data.train_rows} train · ${data.test_rows} test` : undefined} />
        <Stat label="Columns" value={data.columns} hint={`${data.numeric_features.length} numeric · ${data.categorical_features.length} text`} />
        <Stat label="Missing values" value={data.missing_cells} tone={data.missing_cells ? "amber" : "emerald"} hint="filled automatically" />
        <Stat label="Duplicates" value={data.duplicate_rows} tone={data.duplicate_rows ? "amber" : "emerald"} hint="removed" />
        <Stat label="Task" value={isClass ? "Classification" : "Regression"} tone="cyan" hint={`target: ${data.target_column}`} />
        <Stat label="Features used" value={data.used_features.length || "—"} hint="after leakage checks" />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title={isClass ? "Class balance" : "Target distribution"} subtitle={isClass ? "How many rows fall in each class" : `Spread of ${data.target_column}`}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distribution} margin={{ left: 0, right: 8, top: 4, bottom: isClass ? 4 : 40 }}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="name" {...AXIS_PROPS} interval={isClass ? 0 : 2} angle={isClass ? 0 : -35} textAnchor={isClass ? "middle" : "end"} height={isClass ? 30 : 60} />
                <YAxis {...AXIS_PROPS} allowDecimals={false} />
                <Tooltip {...TOOLTIP_PROPS} />
                <Bar dataKey="count" fill={COLORS.violet} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {!isClass && data.target_summary && (
            <p className="mt-2 text-xs text-slate-500">
              Mean {fmt(data.target_summary.mean)} · median {fmt(data.target_summary.median)} · range {fmt(data.target_summary.min)} – {fmt(data.target_summary.max)}
            </p>
          )}
        </Card>

        <Card title="Strongest relationships with the target" subtitle="Correlation of numeric inputs (−1 to +1)">
          {data.relationships.length ? (
            <div style={{ height: Math.max(200, data.relationships.length * 30) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.relationships} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                  <XAxis type="number" domain={[-1, 1]} {...AXIS_PROPS} />
                  <YAxis type="category" dataKey="feature" width={150} {...AXIS_PROPS} />
                  <ReferenceLine x={0} stroke={COLORS.axis} />
                  <Tooltip {...TOOLTIP_PROPS} formatter={(value: number) => [value.toFixed(3), "Correlation"]} />
                  <Bar dataKey="correlation" radius={4}>
                    {data.relationships.map((row) => (
                      <Cell key={row.feature} fill={row.correlation >= 0 ? COLORS.emerald : COLORS.rose} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyHint>Correlations need numeric inputs and a numeric or two-class target.</EmptyHint>
          )}
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Missing values by column" subtitle="Filled from training data only (no leakage)">
          {missing.length ? (
            <div style={{ height: Math.max(160, missing.length * 30) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={missing} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                  <XAxis type="number" {...AXIS_PROPS} unit="%" />
                  <YAxis type="category" dataKey="column" width={150} {...AXIS_PROPS} />
                  <Tooltip {...TOOLTIP_PROPS} formatter={(value: number, _name, item) => [`${pct(value)} (${item.payload.missing} cells)`, "Missing"]} />
                  <Bar dataKey="percent" fill={COLORS.amber} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyHint>No missing values — the dataset is complete.</EmptyHint>
          )}
        </Card>
        <Card title="Data checks" subtitle="From the profile_dataset MCP tool">
          <ul className="space-y-2">
            {data.quality_findings.map((finding) => (
              <li key={finding} className="flex gap-2 text-sm text-slate-300">
                <AlertTriangle size={15} className="mt-0.5 shrink-0 text-amber-300" /> {finding}
              </li>
            ))}
          </ul>
          {data.dataset_hash && (
            <p className="mt-4 flex items-center gap-2 break-all rounded-lg border border-slate-800 bg-slate-950/60 p-2.5 font-mono text-[11px] text-slate-500">
              <Fingerprint size={14} className="shrink-0" /> SHA-256 {data.dataset_hash}
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
