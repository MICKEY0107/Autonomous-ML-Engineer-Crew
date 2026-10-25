import { CheckCircle2 } from "lucide-react";
import type { JsonValue } from "../types";

interface MetricComparisonProps {
  rows: Array<Record<string, JsonValue>>;
}

const PREFERRED_COLUMNS = [
  "name",
  "attempt",
  "metric_name",
  "metric_value",
  "training_time_ms",
  "selected",
];

function labelFor(key: string) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayValue(value: JsonValue | undefined) {
  if (value == null) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  return value;
}

export default function MetricComparison({ rows }: MetricComparisonProps) {
  if (!rows.length) return null;
  const scalarKeys = new Set(
    rows.flatMap((row) => Object.entries(row)
      .filter(([, value]) => value == null || typeof value !== "object")
      .map(([key]) => key)),
  );
  const columns = [
    ...PREFERRED_COLUMNS.filter((key) => scalarKeys.has(key)),
    ...[...scalarKeys].filter((key) => !PREFERRED_COLUMNS.includes(key)),
  ];

  return (
    <div className="reasoning-table-wrap">
      <table className="reasoning-table">
        <thead>
          <tr>{columns.map((column) => <th key={column}>{labelFor(column)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={String(row.run_id ?? row.name ?? index)} className={row.selected === true ? "reasoning-row-selected" : ""}>
              {columns.map((column) => (
                <td key={column}>
                  {column === "selected" && row[column] === true
                    ? <span className="reasoning-selected"><CheckCircle2 size={12} /> Selected</span>
                    : displayValue(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
