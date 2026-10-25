import { memo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

interface MiniChartProps {
  data: number[];
  color: string;
  variant?: "area" | "bar";
}

function MiniChart({ data, color, variant = "area" }: MiniChartProps) {
  const points = data.map((value, index) => ({ index, value }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      {variant === "bar" ? (
        <BarChart data={points} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
          <Tooltip content={() => null} cursor={false} />
          <Bar dataKey="value" fill={color} radius={[2, 2, 0, 0]} isAnimationActive />
        </BarChart>
      ) : (
        <AreaChart data={points} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`fill-${color.slice(1)}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.42} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            fill={`url(#fill-${color.slice(1)})`}
            dot={false}
            isAnimationActive
          />
        </AreaChart>
      )}
    </ResponsiveContainer>
  );
}

export default memo(MiniChart);
