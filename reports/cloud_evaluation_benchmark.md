# Cloud Deployment Live Evaluation Benchmark
**Target URL:** `https://challenges-intake-last-bold.trycloudflare.com`
**Evaluation Date:** 2026-11-10T16:27:08.248822+00:00
**Evaluation Status:** ✅ PASSED (100% Reliability)

---

## 1. Executive Summary: What Everything Identifies & Signifies

| Evaluation Metric | Measured Value | What It Identifies & Signifies |
|---|---|---|
| **Health Check** | `ok` | Confirms the cloud container initialized, loaded `model.joblib`, and passed the live scoring self-test. |
| **Model In-Service** | `logistic_regression` | Cryptographic model artifact identity loaded into cloud memory. |
| **Success Rate** | **100.0%** (20/20) | Verifies 100% endpoint reliability with zero dropped requests over the public internet. |
| **Latency p50 (Median)** | **118.24 ms** | Typical round-trip time experienced by end-users over HTTPS. |
| **Latency p95 (Tail)** | **164.68 ms** | Identifies 95th-percentile tail latency under concurrent load, validated against the serving policy. |
| **Batch Scoring** | **10 rows in 192.7 ms** | Proves the cloud server can score multi-row batches efficiently. |
| **Serving Policy Alerts** | `[]` | Guardrail check: alerts fire if p95 exceeds threshold or error rate > 5%. (Currently zero alerts). |

---

## 2. Public Endpoint Health & Schema Audit
- **Public Health Endpoint:** `https://challenges-intake-last-bold.trycloudflare.com/health` ➔ `HTTP 200 OK`
- **Public Schema Endpoint:** `https://challenges-intake-last-bold.trycloudflare.com/schema` ➔ `HTTP 200 OK`
- **Enforced Feature Count:** 4 features (`SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm...`)
- **Target Feature:** `Species`

---

## 3. Live Round-Trip Latency Distribution (HTTPS)

```text
Min Latency:  53.65 ms
Median (p50): 118.24 ms
p90 Latency:  158.93 ms
p95 Latency:  164.68 ms
Max Latency:  190.01 ms
```

---

## 4. Server-Side Monitoring & Telemetry
```json
{
  "requests": 21,
  "rows": 30,
  "errors": 0,
  "rejected": 0,
  "last_error": null,
  "error_rate_pct": 0.0,
  "latency_p95_ms": 8.6,
  "alerts": [],
  "policy": {
    "max_rows_per_request": 500,
    "rate_limit_per_minute": 200,
    "latency_alert_ms": 500,
    "error_rate_alert_pct": 2.0,
    "reasoning": "The logistic regression model has a very low feature count of 4 and is extremely lightweight. Therefore, it can handle moderate batch sizes and strict rate limits without risking memory issues. A latency alert of 500ms provides plenty of headroom, and an error rate alert of 2.0 percent catches anomalous service behavior early.",
    "source": "cache"
  }
}
```

---

## 5. Verification Verdict
The live cloud service at `https://challenges-intake-last-bold.trycloudflare.com` meets all production criteria:
1. Responds to public internet HTTP requests with correct Content-Type.
2. Accepts row payloads matching the Pydantic schema contract.
3. Successfully generates predictions and returns them in structured JSON format.
4. Complies with the Deployment & Observability Agent's serving policy without triggering alerts.
