import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ClassificationPerformance, RegressionPerformance, ResultPayload } from "../../types";
import { AXIS_PROPS, COLORS, Card, EmptyHint, Stat, TOOLTIP_PROPS, fmt } from "./shared";

function ActualVsPredicted({ perf }: { perf: RegressionPerformance }) {
  const values = perf.scatter.flatMap((point) => [point.actual, point.predicted]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 16 }}>
          <CartesianGrid stroke={COLORS.grid} />
          <XAxis type="number" dataKey="actual" name="Actual" domain={[low, high]} tickFormatter={(v: number) => fmt(v)} {...AXIS_PROPS} label={{ value: "Actual", position: "insideBottom", offset: -8, fill: COLORS.axis, fontSize: 11 }} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[low, high]} tickFormatter={(v: number) => fmt(v)} {...AXIS_PROPS} width={64} />
          <ReferenceLine segment={[{ x: low, y: low }, { x: high, y: high }]} stroke={COLORS.amber} strokeDasharray="5 5" ifOverflow="extendDomain" />
          <Tooltip {...TOOLTIP_PROPS} formatter={(value: number) => fmt(value)} />
          <Scatter data={perf.scatter} fill={COLORS.cyan} fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function Residuals({ perf }: { perf: RegressionPerformance }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={perf.residual_histogram} margin={{ left: 0, right: 8, top: 4, bottom: 40 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="bin" {...AXIS_PROPS} interval={2} angle={-35} textAnchor="end" height={60} />
          <YAxis {...AXIS_PROPS} allowDecimals={false} />
          <Tooltip {...TOOLTIP_PROPS} />
          <Bar dataKey="count" fill={COLORS.violet} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ConfusionMatrix({ perf }: { perf: ClassificationPerformance }) {
  const max = Math.max(1, ...perf.confusion_matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th className="px-2 text-right text-xs font-medium text-slate-500">actual ↓ / predicted →</th>
            {perf.labels.map((label) => (
              <th key={label} className="px-2 py-1 text-xs font-medium text-slate-400">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {perf.confusion_matrix.map((row, i) => (
            <tr key={perf.labels[i]}>
              <th className="px-2 text-right text-xs font-medium text-slate-400">{perf.labels[i]}</th>
              {row.map((count, j) => {
                const intensity = count / max;
                const correct = i === j;
                const color = correct ? `rgba(52,211,153,${0.12 + intensity * 0.6})` : `rgba(251,113,133,${count ? 0.1 + intensity * 0.6 : 0.04})`;
                return (
                  <td key={j} className="h-14 w-20 rounded-md text-center font-mono text-base font-semibold text-slate-100" style={{ background: color }}>
                    {count}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-center text-xs text-slate-500">Green diagonal = correct predictions; red cells = mistakes.</p>
    </div>
  );
}

function PerClass({ perf }: { perf: ClassificationPerformance }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={perf.per_class} margin={{ left: 0, right: 8, top: 4, bottom: 4 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="label" {...AXIS_PROPS} />
          <YAxis domain={[0, 1]} {...AXIS_PROPS} />
          <Tooltip {...TOOLTIP_PROPS} formatter={(value: number) => value.toFixed(3)} />
          <Legend wrapperStyle={{ fontSize: 12, color: COLORS.axis }} />
          <Bar dataKey="precision" fill={COLORS.cyan} radius={[3, 3, 0, 0]} />
          <Bar dataKey="recall" fill={COLORS.violet} radius={[3, 3, 0, 0]} />
          <Bar dataKey="f1" fill={COLORS.emerald} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Roc({ perf }: { perf: ClassificationPerformance }) {
  if (!perf.roc?.length) return <EmptyHint>ROC curves apply to two-class problems.</EmptyHint>;
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={perf.roc} margin={{ left: 0, right: 16, top: 8, bottom: 16 }}>
          <CartesianGrid stroke={COLORS.grid} />
          <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...AXIS_PROPS} label={{ value: "False positive rate", position: "insideBottom", offset: -8, fill: COLORS.axis, fontSize: 11 }} />
          <YAxis type="number" domain={[0, 1]} {...AXIS_PROPS} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={COLORS.slate} strokeDasharray="4 4" />
          <Tooltip {...TOOLTIP_PROPS} formatter={(value: number) => value.toFixed(3)} />
          <Line type="stepAfter" dataKey="tpr" stroke={COLORS.cyan} dot={false} strokeWidth={2} name="True positive rate" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function PerformanceTab({ payload, compact = false }: { payload: ResultPayload; compact?: boolean }) {
  const perf = payload.insights?.performance;
  if (!perf) {
    const empty = <EmptyHint>Performance charts appear once a model has been trained and evaluated.</EmptyHint>;
    return compact ? <Card title="How well it predicts">{empty}</Card> : empty;
  }
  if (perf.kind === "regression") {
    const scatter = (
      <Card title="Predicted vs actual" subtitle="Each dot is an unseen row; closer to the dashed line is better">
        <ActualVsPredicted perf={perf} />
      </Card>
    );
    if (compact) return scatter;
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <Stat label="Average error (MAE)" value={fmt(perf.mae)} tone="cyan" />
          <Stat label="RMSE" value={fmt(perf.rmse)} />
          <Stat label="Typical error" value={fmt(perf.median_abs_error)} hint="median absolute error" />
          <Stat label="Within 10%" value={`${fmt(perf.within_10_percent)}%`} tone="emerald" hint="of predictions" />
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          {scatter}
          <Card title="Error distribution" subtitle="Actual − predicted; centred on zero means no systematic bias">
            <Residuals perf={perf} />
          </Card>
        </div>
      </div>
    );
  }
  const matrix = (
    <Card title="Confusion matrix" subtitle="Where the model is right and where it mixes classes up">
      <ConfusionMatrix perf={perf} />
    </Card>
  );
  if (compact) return matrix;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label="Accuracy" value={`${fmt(perf.accuracy * 100)}%`} tone="emerald" />
        {perf.per_class.slice(0, 3).map((row) => (
          <Stat key={row.label} label={`Recall · ${row.label}`} value={fmt(row.recall)} hint={`${row.support} rows`} />
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {matrix}
        <Card title="Quality per class" subtitle="Precision, recall and F1 on unseen data">
          <PerClass perf={perf} />
        </Card>
      </div>
      <Card title="ROC curve" subtitle={perf.positive_label ? `Positive class: ${perf.positive_label}` : undefined}>
        <Roc perf={perf} />
      </Card>
    </div>
  );
}
