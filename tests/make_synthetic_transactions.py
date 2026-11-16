"""Generate data/sample_transactions.csv - a SYNTHETIC demo file.

It is deliberately shaped unlike the retail dataset, to demo the column mapper
and the RFM path on page 4:
  * many rows per customer, with dates (about 2,000 customers, ~15k rows,
    ~18 months);
  * different column names (cust_id, invoice_no, invoice_date, order_total,
    product_group, sex, promo);
  * a few bad rows: unreadable dates, non-numeric totals, refunds (< 0).

Customer behaviour types (frequent/recent, regular, lapsed, one-off) are mixed
in so that RFM segments exist to be found. All values are random; the fixed
seed makes the file identical on every run.

Run:  python -m tests.make_synthetic_transactions
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

SEED = 7
N_CUSTOMERS = 2_000
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-06-30")
OUT_PATH = config.DATA_DIR / "sample_transactions.csv"

# type: (share of customers, mean orders, mean order value, active window)
TYPES = {
    "frequent": (0.15, 25, 70, (0.0, 1.0)),
    "regular": (0.35, 8, 45, (0.0, 1.0)),
    "lapsed": (0.25, 5, 50, (0.0, 0.45)),
    "one_off": (0.25, 1.3, 35, (0.0, 1.0)),
}
GROUPS = ["Apparel", "Shoes", "Bags", "Jewellery", "Outdoor"]


def make() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    span_days = (END - START).days
    kinds = rng.choice(list(TYPES), size=N_CUSTOMERS,
                       p=[t[0] for t in TYPES.values()])
    rows = []
    invoice = 100_000
    for cid, kind in enumerate(kinds, start=1):
        _, mean_orders, mean_value, (lo, hi) = TYPES[kind]
        n = max(1, rng.poisson(mean_orders))
        sex = rng.choice(["M", "F"])
        fav = rng.choice(GROUPS)
        day = rng.integers(int(lo * span_days), int(hi * span_days) + 1, size=n)
        for d in np.sort(day):
            invoice += 1
            rows.append({
                "cust_id": f"C{cid:05d}",
                "invoice_no": f"INV{invoice}",
                "invoice_date": (START + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"),
                "order_total": round(float(rng.gamma(4, mean_value / 4)), 2),
                "product_group": fav if rng.random() < 0.6 else rng.choice(GROUPS),
                "sex": sex,
                "promo": rng.choice(["Y", "N"], p=[0.3, 0.7]),
            })
    df = pd.DataFrame(rows)

    # A few deliberately bad rows.
    bad = rng.choice(len(df), size=30, replace=False)
    df.loc[bad[:10], "invoice_date"] = "not a date"
    df["order_total"] = df["order_total"].astype(object)
    df.loc[bad[10:20], "order_total"] = "n/a"
    df.loc[bad[20:], "order_total"] = -df.loc[bad[20:], "order_total"].astype(float)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    data = make()
    data.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(data):,} rows for {data['cust_id'].nunique():,} customers "
          f"to {OUT_PATH}")
