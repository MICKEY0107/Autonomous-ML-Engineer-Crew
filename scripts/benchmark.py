"""End-to-end benchmark: repeated pipeline runs -> runtime, cost and quality.

Usage (from the project root):
    .venv\\Scripts\\python.exe scripts\\benchmark.py --runs 3
    .venv\\Scripts\\python.exe scripts\\benchmark.py --runs 5 --offline

Each run is a real ``main.py`` execution (A2A server, MCP tools, MLflow, LLM
decisions). Results are read from each run's saved ``pipeline_result.json``
and summarised as p50/p95 runtime, success rate, tokens and cost per run in
``reports/benchmark.md`` and ``reports/benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("data/house_prices.csv", "sale_price", "predict the sale price of a house"),
    ("data/customer_churn.csv", "churned", "predict which customers will churn"),
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def run_once(csv: str, target: str, goal: str, env: dict[str, str]) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "main.py", "--csv", csv, "--target", target, "--goal", goal],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900,
    )
    wall = time.perf_counter() - started
    match = re.search(r"\[RESULT\]\s+(.+pipeline_result\.json)", completed.stdout)
    if not match:
        return {"dataset": csv, "status": "error", "wall_seconds": wall,
                "error": (completed.stdout + completed.stderr)[-400:]}
    result = json.loads((ROOT / match.group(1).strip()).read_text(encoding="utf-8"))
    usage = result.get("usage_report") or {}
    evaluation = result.get("evaluation") or {}
    metric = evaluation.get("primary_metric")
    return {
        "dataset": csv,
        "status": result.get("status"),
        "passed": bool(evaluation.get("passed")),
        "runtime_seconds": result.get("runtime_seconds"),
        "wall_seconds": round(wall, 2),
        "metric": metric,
        "metric_value": (evaluation.get("held_out_metrics") or {}).get(metric),
        "tokens": (result.get("input_tokens") or 0) + (result.get("output_tokens") or 0),
        "cost_usd": result.get("estimated_cost_usd") or 0.0,
        "llm_calls": usage.get("llm_requests", 0),
        "local_calls": usage.get("local_llm_requests", 0),
        "cache_hits": usage.get("cache_hits", 0),
        "fallbacks": usage.get("fallback_decisions", 0),
        "deployed": result.get("deployment") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="LLM_MODE=offline (no LLM calls)")
    parser.add_argument("--no-cache", action="store_true", help="LLM_CACHE=off (every decision live)")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if args.offline:
        env["LLM_MODE"] = "offline"
    if args.no_cache:
        env["LLM_CACHE"] = "off"

    results = []
    for csv, target, goal in DATASETS:
        for attempt in range(args.runs):
            print(f"[{csv}] run {attempt + 1}/{args.runs} ...", flush=True)
            results.append(run_once(csv, target, goal, env))
            print("   ", {k: results[-1].get(k) for k in ("status", "runtime_seconds", "cost_usd", "llm_calls")}, flush=True)

    summary = {}
    for csv, _, _ in DATASETS:
        rows = [r for r in results if r["dataset"] == csv]
        runtimes = [r["runtime_seconds"] for r in rows if r.get("runtime_seconds")]
        summary[csv] = {
            "runs": len(rows),
            "success_rate_percent": round(100 * sum(r.get("passed", False) for r in rows) / max(len(rows), 1), 1),
            "runtime_p50_s": round(_percentile(runtimes, 50), 2),
            "runtime_p95_s": round(_percentile(runtimes, 95), 2),
            "mean_cost_usd": round(statistics.fmean([r.get("cost_usd", 0) for r in rows]) if rows else 0, 6),
            "mean_tokens": round(statistics.fmean([r.get("tokens", 0) for r in rows]) if rows else 0),
            "mean_llm_calls": round(statistics.fmean([r.get("llm_calls", 0) for r in rows]) if rows else 0, 1),
            "mean_metric": round(statistics.fmean([r["metric_value"] for r in rows if r.get("metric_value") is not None]) if rows else 0, 4),
            "metric": next((r.get("metric") for r in rows if r.get("metric")), None),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "benchmark.json").write_text(json.dumps({"generated": stamp, "summary": summary, "runs": results}, indent=2), encoding="utf-8")
    lines = [
        "# End-to-end benchmark",
        "",
        f"Generated {stamp} · {args.runs} runs per dataset · "
        f"LLM_MODE={env.get('LLM_MODE', 'live')} · LLM_CACHE={env.get('LLM_CACHE', 'on')} · "
        f"primary model {env.get('CREWAI_MODEL', '?')}",
        "",
        "| Dataset | Runs | Pass rate | Runtime p50 | Runtime p95 | Mean LLM calls | Mean tokens | Mean cost / run | Mean held-out metric |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for csv, row in summary.items():
        lines.append(
            f"| {Path(csv).name} | {row['runs']} | {row['success_rate_percent']}% | {row['runtime_p50_s']} s | "
            f"{row['runtime_p95_s']} s | {row['mean_llm_calls']} | {row['mean_tokens']} | ${row['mean_cost_usd']:.6f} | "
            f"{row['metric']} {row['mean_metric']} |"
        )
    lines += ["", "Raw per-run data: `reports/benchmark.json`.", ""]
    (reports / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
