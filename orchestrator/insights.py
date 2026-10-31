"""Chart-ready insights for the dashboard, computed locally after a run.

Everything here is computed in-process from the run's own artifacts (the
uploaded CSV, the held-out split and the registered model). It is shown to the
person who uploaded the data and saved next to the run; it is never sent to an
LLM prompt or a trace span.

Payload sections:
* ``dataset``      - size, column types, missing values, target distribution,
                     strongest numeric relationships with the target, leakage.
* ``performance``  - regression: actual vs predicted, residuals, error stats;
                     classification: confusion matrix, per-class scores, ROC.
* ``importance``   - permutation importance on the held-out split (model-agnostic,
                     reported on the original input columns).
* ``findings``     - plain-language bullet points for a non-expert reader.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_POINTS = 300
MAX_IMPORTANCE_ROWS = 2_000


def _finite(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _short(value: float) -> str:
    """Human-readable axis label: 1.2M, 460k, 3.5, -12k."""

    magnitude = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"


def _histogram(values: pd.Series, bins: int = 20) -> list[dict[str, Any]]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return []
    counts, edges = np.histogram(clean, bins=min(bins, max(clean.nunique(), 1)))
    return [
        {
            "bin": f"{_short(edges[i])} – {_short(edges[i + 1])}",
            "start": _finite(edges[i]),
            "end": _finite(edges[i + 1]),
            "count": int(count),
        }
        for i, count in enumerate(counts)
    ]


# --------------------------------------------------------------------------- data
def dataset_insights(
    csv_path: str | Path,
    target_column: str,
    task_type: str,
    data_prep: Any | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    target = frame[target_column]
    features = frame.drop(columns=[target_column])
    numeric = features.select_dtypes(include="number")
    categorical = [c for c in features.columns if c not in numeric.columns]

    missing = [
        {
            "column": column,
            "missing": int(count),
            "percent": _finite(count / max(len(frame), 1) * 100, 2),
        }
        for column, count in frame.isna().sum().sort_values(ascending=False).items()
    ]

    if task_type == "classification":
        counts = target.astype(str).value_counts()
        target_distribution = [
            {"label": str(label), "count": int(count), "percent": _finite(count / len(target) * 100, 2)}
            for label, count in counts.items()
        ]
    else:
        target_distribution = _histogram(target)

    relationships: list[dict[str, Any]] = []
    if task_type != "classification" or target.nunique() == 2:
        numeric_target = (
            pd.to_numeric(target, errors="coerce")
            if task_type != "classification"
            else pd.Series(pd.factorize(target)[0], index=target.index)
        )
        for column in numeric.columns:
            correlation = numeric[column].corr(numeric_target)
            if correlation is not None and math.isfinite(correlation):
                relationships.append(
                    {"feature": column, "correlation": _finite(correlation, 4)}
                )
        relationships.sort(key=lambda item: abs(item["correlation"] or 0), reverse=True)

    # Classification: how each class differs on the most separating inputs
    # (average value per class, scaled 0-100 so features share one axis).
    class_profiles: list[dict[str, Any]] = []
    class_labels: list[str] = []
    if task_type == "classification" and not numeric.empty:
        model_features = set(getattr(data_prep, "feature_columns", None) or numeric.columns)
        usable = [c for c in numeric.columns if c in model_features] or list(numeric.columns)
        grouped = numeric[usable].groupby(target.astype(str)).mean()
        spread = (grouped.max() - grouped.min()) / (numeric[usable].std(ddof=0) + 1e-12)
        top = list(spread.sort_values(ascending=False).index[:6])
        class_labels = [str(label) for label in grouped.index]
        for column in top:
            low, high = numeric[column].min(), numeric[column].max()
            row: dict[str, Any] = {"feature": column}
            for label in class_labels:
                value = grouped.loc[label, column]
                row[label] = _finite((value - low) / (high - low) * 100 if high > low else 50, 2)
                row[f"{label}__raw"] = _finite(value, 4)
            class_profiles.append(row)

    target_summary: dict[str, Any] = {}
    if task_type != "classification":
        numeric_target = pd.to_numeric(target, errors="coerce")
        target_summary = {
            "mean": _finite(numeric_target.mean(), 4),
            "median": _finite(numeric_target.median(), 4),
            "std": _finite(numeric_target.std(), 4),
            "min": _finite(numeric_target.min(), 4),
            "max": _finite(numeric_target.max(), 4),
        }

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "numeric_features": list(numeric.columns),
        "categorical_features": categorical,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_by_column": missing,
        "target_column": target_column,
        "task_type": task_type,
        "target_distribution": target_distribution,
        "target_summary": target_summary,
        "relationships": relationships[:12],
        "class_profiles": class_profiles,
        "class_labels": class_labels,
        "dataset_hash": getattr(data_prep, "dataset_hash", None),
        "quality_findings": list(getattr(data_prep, "quality_findings", []) or []),
        "train_rows": getattr(data_prep, "train_row_count", None),
        "test_rows": getattr(data_prep, "test_row_count", None),
        "used_features": list(getattr(data_prep, "feature_columns", []) or []),
    }


# -------------------------------------------------------------------- performance
def _regression_performance(target: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    actual = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    residuals = actual - predictions
    order = np.random.default_rng(42).permutation(len(actual))[:MAX_POINTS]
    return {
        "kind": "regression",
        "scatter": [
            {"actual": _finite(actual[i], 4), "predicted": _finite(predictions[i], 4)}
            for i in order
        ],
        "residual_histogram": _histogram(pd.Series(residuals)),
        "mae": _finite(np.mean(np.abs(residuals)), 4),
        "rmse": _finite(np.sqrt(np.mean(residuals**2)), 4),
        "median_abs_error": _finite(np.median(np.abs(residuals)), 4),
        "within_10_percent": _finite(
            np.mean(np.abs(residuals) <= 0.10 * np.maximum(np.abs(actual), 1e-12)) * 100, 2
        ),
    }


def _classification_performance(model: Any, features: pd.DataFrame, target: pd.Series, predictions: Any) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve

    labels = sorted(pd.unique(pd.concat([target, pd.Series(predictions)]).astype(str)))
    actual = target.astype(str).to_numpy()
    predicted = pd.Series(predictions).astype(str).to_numpy()
    matrix = confusion_matrix(actual, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    payload: dict[str, Any] = {
        "kind": "classification",
        "labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": [
            {
                "label": label,
                "precision": _finite(precision[i], 4),
                "recall": _finite(recall[i], 4),
                "f1": _finite(f1[i], 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        ],
        "accuracy": _finite((actual == predicted).mean(), 4),
    }
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba) and len(labels) == 2:
        try:
            probabilities = predict_proba(features)[:, 1]
            positive = str(getattr(model, "classes_", labels)[-1])
            fpr, tpr, _ = roc_curve(actual == positive, probabilities)
            step = max(1, len(fpr) // 100)
            payload["roc"] = [
                {"fpr": _finite(fpr[i], 4), "tpr": _finite(tpr[i], 4)}
                for i in range(0, len(fpr), step)
            ] + [{"fpr": 1.0, "tpr": 1.0}]
            payload["positive_label"] = positive
        except (TypeError, ValueError, IndexError):
            pass
    return payload


def _importance(model: Any, features: pd.DataFrame, target: pd.Series, task_type: str) -> list[dict[str, Any]]:
    from sklearn.inspection import permutation_importance

    sample_x, sample_y = features, target
    if len(features) > MAX_IMPORTANCE_ROWS:
        sample_x = features.sample(MAX_IMPORTANCE_ROWS, random_state=42)
        sample_y = target.loc[sample_x.index]
    scoring = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
    try:
        result = permutation_importance(
            model, sample_x, sample_y, n_repeats=5, random_state=42, scoring=scoring
        )
    except Exception:
        return []
    rows = [
        {
            "feature": column,
            "importance": _finite(result.importances_mean[i], 6),
            "std": _finite(result.importances_std[i], 6),
        }
        for i, column in enumerate(sample_x.columns)
    ]
    rows.sort(key=lambda item: item["importance"] or 0, reverse=True)
    total = sum(max(item["importance"] or 0, 0) for item in rows) or 1.0
    for item in rows:
        item["share_percent"] = _finite(max(item["importance"] or 0, 0) / total * 100, 2)
    return rows[:12]


def model_insights(selection: Any, tracking_uri: str, artifact_dir: str | Path, task_type: str) -> dict[str, Any]:
    from mlflow import MlflowClient

    from agents.evaluation import _download_evaluation_bundle

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    model, held_out, metadata = _download_evaluation_bundle(
        client, selection, Path(artifact_dir) / "insights-cache"
    )
    features = held_out[metadata.feature_columns]
    target = held_out[metadata.target_column]
    predictions = model.predict(features)
    if task_type == "classification":
        performance = _classification_performance(model, features, target, predictions)
    else:
        performance = _regression_performance(target, np.asarray(predictions, dtype=float))
    return {
        "performance": performance,
        "importance": _importance(model, features, target, task_type),
    }


# ----------------------------------------------------------------------- findings
def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.3g}" if abs(number) < 1 else f"{number:,.2f}"


def _exhausted_note(attempts: list[dict[str, Any]], last_model: str) -> str:
    """Name the attempt that came closest to the bar, not just the last one."""

    scored = []
    for item in attempts:
        evaluation = item.get("evaluation") or {}
        metric = evaluation.get("primary_metric")
        value = (evaluation.get("held_out_metrics") or {}).get(metric)
        if metric and value is not None:
            scored.append((float(value), item, evaluation))
    if not scored:
        return (
            f"No model reached the quality bar after {len(attempts)} attempts; "
            f"the last one tried was {last_model}."
        )
    lower_is_better = "min" in str(scored[0][2].get("direction", "")).casefold()
    value, best, evaluation = (min if lower_is_better else max)(scored, key=lambda row: row[0])
    name = str(best.get("selected_model") or last_model).replace("_", " ")
    note = (
        f"No model reached the quality bar after {len(attempts)} attempts; the closest was "
        f"{name} on attempt {best.get('attempt')} ({evaluation.get('primary_metric')} "
        f"{_fmt(value)} vs target {_fmt(evaluation.get('threshold'))})."
    )
    if best is not attempts[-1]:
        note += " The charts below describe the final attempt."
    return note


def findings(insights: dict[str, Any], final: dict[str, Any]) -> list[str]:
    data = insights.get("dataset", {})
    perf = insights.get("performance", {})
    importance = insights.get("importance", [])
    attempts = insights.get("attempts", [])
    notes: list[str] = []
    status = final.get("status")
    model = (final.get("model_name") or "the model").replace("_", " ")
    if status == "SUCCESS":
        notes.append(f"A {model} model passed the held-out quality check and is ready to use.")
    elif status == "FAILED_EVAL_EXHAUSTED":
        notes.append(_exhausted_note(attempts, model))
    if perf.get("kind") == "regression":
        notes.append(
            f"On unseen data, predictions are off by {_fmt(perf.get('mae'))} on average "
            f"(typical error {_fmt(perf.get('median_abs_error'))}); "
            f"{_fmt(perf.get('within_10_percent'))}% land within 10% of the true value."
        )
    elif perf.get("kind") == "classification":
        notes.append(f"On unseen data the model is right {_fmt((perf.get('accuracy') or 0) * 100)}% of the time.")
        weakest = min(perf.get("per_class", []), key=lambda r: r.get("recall") or 0, default=None)
        if weakest and (weakest.get("recall") or 1) < 0.8:
            notes.append(
                f"Class '{weakest['label']}' is the hardest to catch (recall {_fmt(weakest['recall'])})."
            )
    if importance:
        top = [item for item in importance[:3] if (item.get("importance") or 0) > 0]
        if top:
            names = ", ".join(f"{t['feature']} ({_fmt(t['share_percent'])}%)" for t in top)
            notes.append(f"Most influential inputs: {names}.")
    if data.get("missing_cells"):
        worst = data["missing_by_column"][0]
        notes.append(
            f"{data['missing_cells']} missing values were filled automatically "
            f"(most in '{worst['column']}', {_fmt(worst['percent'])}%)."
        )
    if data.get("duplicate_rows"):
        notes.append(f"{data['duplicate_rows']} duplicate rows were removed before training.")
    if len(attempts) > 1:
        notes.append(
            f"The evaluator sent the model back {len(attempts) - 1} time(s) with feedback before the final result."
        )
    return notes


# --------------------------------------------------------------------- model card
def model_card_markdown(final: dict[str, Any], insights: dict[str, Any]) -> str:
    data = insights.get("dataset", {})
    perf = insights.get("performance", {})
    lines = [
        "# Model card — Autonomous ML Engineer Crew",
        "",
        f"- **Run ID:** `{final.get('run_id')}`",
        f"- **Status:** {final.get('status')}",
        f"- **Task:** {data.get('task_type')} on target `{data.get('target_column')}`",
        f"- **Prediction goal:** {final.get('prediction_goal', '')}",
        f"- **Selected model:** {final.get('model_name')}",
        f"- **Held-out {final.get('metric_name')}:** {_fmt(final.get('metric_value'))} "
        f"(gate {_fmt(final.get('threshold'))})",
        f"- **Attempts:** {len(insights.get('attempts', [])) or 1} · retries {final.get('retry_count', 0)}",
        f"- **Runtime:** {(final.get('runtime_ms') or 0) / 1000:.1f}s · "
        f"LLM cost ${final.get('estimated_cost_usd', 0):.6f}",
        f"- **Dataset:** {data.get('rows')} rows × {data.get('columns')} columns · SHA-256 `{data.get('dataset_hash')}`",
        "",
        "## Key findings",
        "",
        *[f"- {note}" for note in insights.get("findings", [])],
        "",
        "## Most influential inputs",
        "",
        "| Feature | Share of importance |",
        "| --- | --- |",
        *[f"| {i['feature']} | {_fmt(i.get('share_percent'))}% |" for i in insights.get("importance", [])[:8]],
        "",
    ]
    if perf.get("kind") == "classification":
        lines += ["## Per-class quality", "", "| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"]
        lines += [
            f"| {r['label']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | {r['support']} |"
            for r in perf.get("per_class", [])
        ]
    elif perf.get("kind") == "regression":
        lines += [
            "## Error profile",
            "",
            f"- MAE {_fmt(perf.get('mae'))} · RMSE {_fmt(perf.get('rmse'))} · median absolute error {_fmt(perf.get('median_abs_error'))}",
            f"- {_fmt(perf.get('within_10_percent'))}% of predictions within 10% of the actual value",
        ]
    lines += [
        "",
        "## How decisions were made",
        "",
        "| Stage | Agent | Source | Model | Tokens | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for call in (final.get("usage_report") or {}).get("calls", []):
        lines.append(
            f"| {call.get('stage')} | {call.get('agent')} | {call.get('source')} | "
            f"{call.get('model') or '—'} | {call.get('total_tokens', 0)} | {call.get('estimated_cost_usd', 0):.6f} |"
        )
    lines += [
        "",
        f"Trace: {final.get('trace_url')}",
        "",
        "_Raw dataset rows were never sent to an LLM or a trace; LLMs saw schema, counts and metrics only._",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- build
def build_run_insights(
    *,
    state: Any,
    attempts: list[dict[str, Any]],
    tracking_uri: str,
    artifact_dir: str | Path,
    final: dict[str, Any],
) -> dict[str, Any]:
    """Compute everything the results page needs; failures degrade per section."""

    from orchestrator.flow import _local_path  # noqa: PLC0415

    payload: dict[str, Any] = {"attempts": attempts, "errors": []}
    try:
        payload["dataset"] = dataset_insights(
            _local_path(state.dataset_uri),
            state.target_column,
            str(state.task_type),
            state.data_prep_result,
        )
    except Exception as exc:  # a chart must never fail the run
        payload["errors"].append(f"dataset: {type(exc).__name__}")
    if state.model_selection_result is not None:
        try:
            payload.update(
                model_insights(
                    state.model_selection_result, tracking_uri, artifact_dir, str(state.task_type)
                )
            )
        except Exception as exc:
            payload["errors"].append(f"model: {type(exc).__name__}")
    payload["findings"] = findings(payload, final)
    payload["model_card_markdown"] = model_card_markdown(final, payload)
    target = Path(artifact_dir) / "insights.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except OSError:
        pass
    return payload
