"""Generate the bundled classification demo dataset (customer churn).

Deterministic (seeded) synthetic data with realistic signal: short tenure,
month-to-month contracts, many support calls and high monthly charges raise
churn risk. A few missing values are included on purpose.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

ROWS = 800
rng = random.Random(7)
records = []
for index in range(ROWS):
    tenure = rng.randint(1, 72)
    contract = rng.choices(["month-to-month", "one-year", "two-year"], [0.55, 0.25, 0.20])[0]
    monthly = round(rng.uniform(20, 120), 2)
    calls = min(int(rng.expovariate(0.6)), 10)
    internet = rng.choices(["fiber", "dsl", "none"], [0.45, 0.40, 0.15])[0]
    paperless = rng.random() < 0.6
    senior = rng.random() < 0.16
    score = (
        -1.2
        - 0.045 * tenure
        + {"month-to-month": 1.6, "one-year": 0.2, "two-year": -1.0}[contract]
        + 0.018 * (monthly - 65)
        + 0.35 * calls
        + (0.5 if internet == "fiber" else 0.0)
        + (0.3 if senior else 0.0)
        + rng.gauss(0, 0.6)
    )
    churned = "yes" if 1 / (1 + math.exp(-score)) > 0.5 else "no"
    records.append(
        {
            "customer_id": f"C{index + 10000}",
            "tenure_months": tenure,
            "contract": contract,
            "monthly_charges": monthly,
            "total_charges": round(monthly * tenure * rng.uniform(0.95, 1.05), 2),
            "support_calls": calls,
            "internet_service": internet,
            "paperless_billing": "yes" if paperless else "no",
            "senior_citizen": "yes" if senior else "no",
            "churned": churned,
        }
    )
frame = pd.DataFrame(records)
for column in ("total_charges", "internet_service"):
    frame.loc[frame.sample(12, random_state=len(column)).index, column] = None
target = Path(__file__).resolve().parents[1] / "data" / "customer_churn.csv"
frame.to_csv(target, index=False)
print(f"wrote {target} ({len(frame)} rows, churn rate {(frame.churned == 'yes').mean():.1%})")
