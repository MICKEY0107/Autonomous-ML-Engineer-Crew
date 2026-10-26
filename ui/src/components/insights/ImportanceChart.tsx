import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Insights } from "../../types";
import { AXIS_PROPS, COLORS, EmptyHint, TOOLTIP_PROPS } from "./shared";

export default function ImportanceChart({ importance }: { importance: Insights["importance"] }) {
  const rows = (importance ?? []).filter((row) => (row.share_percent ?? 0) > 0).slice(0, 10);
  if (!rows.length) return <EmptyHint>Feature importance is not available for this run.</EmptyHint>;
  return (
    <div style={{ height: Math.max(180, rows.length * 34) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" {...AXIS_PROPS} unit="%" />
          <YAxis type="category" dataKey="feature" width={150} {...AXIS_PROPS} />
          <Tooltip {...TOOLTIP_PROPS} formatter={(value: number) => [`${value.toFixed(1)}%`, "Share of importance"]} />
          <Bar dataKey="share_percent" radius={[0, 4, 4, 0]}>
            {rows.map((row, index) => (
              <Cell key={row.feature} fill={index === 0 ? COLORS.cyan : "#0e7490"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
