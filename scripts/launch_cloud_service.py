"""Public Cloud Service Launcher & Evaluator.

Starts the standalone prediction container service (serving/app.py) and
exposes it to the public internet with a live HTTPS URL via Cloudflare's
global edge network. Then automatically runs the cloud evaluation benchmark.

Usage::

    .venv\\Scripts\\python.exe scripts\\launch_cloud_service.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUDFLARED = PROJECT_ROOT / "scripts" / "cloudflared.exe"
SERVING_DIR = PROJECT_ROOT / "serving"
DEFAULT_BUNDLE = PROJECT_ROOT / "artifacts" / "deployments" / "11aba26804f04096abc065a7ca06668d"


def launch_public_cloud_service(bundle_dir: Path = DEFAULT_BUNDLE, port: int = 8990) -> None:
    if not CLOUDFLARED.is_file():
        raise FileNotFoundError(f"cloudflared not found at {CLOUDFLARED}")
    if not (bundle_dir / "model.joblib").is_file():
        raise FileNotFoundError(f"Model bundle not found at {bundle_dir}")

    print("=====================================================================")
    print("🚀 LAUNCHING AUTONOMOUS ML ENGINEER CREW — LIVE PUBLIC CLOUD SERVICE")
    print("=====================================================================")
    print(f"📦 Model Bundle: {bundle_dir.name}")
    print(f"🔌 Local Port   : {port}\n")

    # 1. Start prediction service
    env = os.environ.copy()
    env["MODEL_DIR"] = str(bundle_dir)
    env["PORT"] = str(port)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    print("  [1/3] Starting prediction engine …", end=" ", flush=True)
    app_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2)
    if app_proc.poll() is not None:
        out, _ = app_proc.communicate()
        print(f"FAILED!\n{out}")
        return
    print("ONLINE ✓")

    # 2. Start Cloudflare Tunnel to expose public HTTPS URL
    print("  [2/3] Provisioning public HTTPS edge tunnel …", end=" ", flush=True)
    tunnel_proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    public_url = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = tunnel_proc.stdout.readline()
        if not line and tunnel_proc.poll() is not None:
            break
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            public_url = match.group(0)
            break
        time.sleep(0.1)

    if not public_url:
        print("FAILED to get tunnel URL.")
        app_proc.terminate()
        tunnel_proc.terminate()
        return

    print("CONNECTED ✓")
    print(f"\n🌐 ===================================================================")
    print(f"🌐 LIVE PUBLIC CLOUD URL: {public_url}")
    print(f"🌐 Health Check: {public_url}/health")
    print(f"🌐 Schema Doc  : {public_url}/schema")
    print(f"🌐 Predict API : {public_url}/predict")
    print(f"🌐 Metrics API : {public_url}/metrics")
    print(f"🌐 ===================================================================\n")

    # 3. Run live evaluation benchmark
    print("  [3/3] Running automated cloud evaluation benchmark against live URL …\n")
    from scripts.test_cloud_live import evaluate_cloud_endpoint
    benchmark = evaluate_cloud_endpoint(public_url, test_csv=PROJECT_ROOT / "data" / "iris.csv")

    print("\n=====================================================================")
    print("🎉 CLOUD EVALUATION COMPLETE — EVIDENCE RECORDED")
    print(f"   Success Rate : {benchmark['success_rate_pct']}%")
    print(f"   Latency p50  : {benchmark['latency_p50_ms']} ms")
    print(f"   Latency p95  : {benchmark['latency_p95_ms']} ms")
    print(f"   Report Path  : reports/cloud_evaluation_benchmark.md")
    print("=====================================================================")
    print("\nThe cloud service is running live. Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping cloud service …")
        tunnel_proc.terminate()
        app_proc.terminate()


if __name__ == "__main__":
    launch_public_cloud_service()
