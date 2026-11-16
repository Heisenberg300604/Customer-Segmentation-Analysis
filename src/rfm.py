"""RFM: turn transaction rows (many per customer) into one row per customer.

  recency_days    days between the customer's last purchase and the snapshot
                  date (default: the day after the last date in the file)
  frequency       number of distinct orders (order id if mapped, else
                  distinct purchase days)
  monetary        total amount spent
  avg_order_value monetary / frequency
  tenure_days     days between first and last purchase

Low recency + high frequency + high monetary = the best customers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

RFM_FEATURES = ["recency_days", "frequency", "monetary"]
# Frequency and monetary are right-skewed (a few huge customers); log1p
# compresses that tail before scaling so K-Means is not dominated by outliers.
RFM_LOG_FEATURES = ["frequency", "monetary"]
MIN_RANGE_DAYS = 30


@dataclass
class RFMResult:
    df: pd.DataFrame
    snapshot_date: pd.Timestamp
    report: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def can_build_rfm(schema: dict) -> bool:
    """RFM needs a customer id, a date and an amount."""
    return all(r in schema for r in ("customer_id", "transaction_date", "amount"))


def _mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if not m.empty else None


def build_rfm(
    df: pd.DataFrame,
    snapshot_date=None,
    dayfirst: bool = False,
    exclude_nonpositive: bool = True,
) -> RFMResult:
    """Build the customer-level RFM table from canonical transaction rows.

    Args:
        df: canonical DataFrame with customer_id, transaction_date,
            purchase_amount (and optionally order_id, category, gender, ...).
        snapshot_date: "today" for recency. Default = max date + 1 day.
        dayfirst: used only if dates are still text.
        exclude_nonpositive: drop refunds/returns (amount <= 0) first.
    """
    report, warnings = [], []
    tx = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(tx["transaction_date"]):
        tx["transaction_date"] = pd.to_datetime(
            tx["transaction_date"], errors="coerce", format="mixed",
            dayfirst=dayfirst)
    bad_date = tx["transaction_date"].isna()
    if bad_date.any():
        report.append(f"Ignored {int(bad_date.sum())} row(s) with an unreadable date.")
        tx = tx[~bad_date]
    no_id = tx["customer_id"].isna()
    if no_id.any():
        report.append(f"Ignored {int(no_id.sum())} row(s) without a customer id.")
        tx = tx[~no_id]
    if exclude_nonpositive:
        refunds = tx["purchase_amount"] <= 0
        if refunds.any():
            report.append(f"Excluded {int(refunds.sum())} refund/return row(s) "
                          "(amount ≤ 0).")
            tx = tx[~refunds]
    if tx.empty:
        raise ValueError("No usable transactions left after cleaning.")

    first, last = tx["transaction_date"].min(), tx["transaction_date"].max()
    span = (last - first).days
    if span < MIN_RANGE_DAYS:
        warnings.append(f"The file covers only {span} day(s); recency and "
                        "frequency are not very meaningful over such a short period.")
    snapshot = (pd.Timestamp(snapshot_date) if snapshot_date is not None
                else last.normalize() + pd.Timedelta(days=1))

    tx["_order_key"] = (tx["order_id"].astype(str) if "order_id" in tx.columns
                        else tx["transaction_date"].dt.normalize().astype(str))
    g = tx.groupby("customer_id")
    rfm = pd.DataFrame({
        "last_purchase": g["transaction_date"].max(),
        "first_purchase": g["transaction_date"].min(),
        "frequency": g["_order_key"].nunique(),
        "monetary": g["purchase_amount"].sum(),
    })
    rfm["recency_days"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

    # Customer-level attributes for the dashboard.
    for col in ["category", "gender", "location", "subscription_status",
                "shipping_type", "age_group"]:
        if col in tx.columns:
            rfm[col] = g[col].agg(_mode)
    if "age" in tx.columns:
        rfm["age"] = g["age"].median()
    if "discount_flag" in tx.columns:
        rfm["discount_share"] = g["discount_flag"].mean()

    rfm = rfm.reset_index()
    report.append(f"Built {len(rfm):,} customer rows from {len(tx):,} "
                  f"transactions ({first.date()} to {last.date()}); "
                  f"recency measured from {snapshot.date()}.")
    return RFMResult(df=rfm, snapshot_date=snapshot, report=report,
                     warnings=warnings)
