"""Tests for src/rfm.py on a tiny dataset checked by hand."""
import pandas as pd
import pytest

from src import mapping, rfm


@pytest.fixture
def tiny():
    # 3 customers, 9 rows (one refund, one bad date).
    return pd.DataFrame({
        "customer": ["A", "A", "A", "B", "B", "C", "C", "C", "A"],
        "date": ["2025-01-01", "2025-01-15", "2025-03-01",
                 "2025-02-01", "2025-02-01",
                 "2025-01-10", "2025-03-10", "bad", "2025-03-05"],
        "total": [10, 20, 30, 50, 25, 5, 15, 99, -20],
    })


def _canon(df):
    m = {r: None for r in mapping.ROLES} | {
        "customer_id": "customer", "transaction_date": "date", "amount": "total"}
    return mapping.apply_mapping(df, m)


def test_bad_date_reported(tiny):
    md = _canon(tiny)
    assert any("unreadable date" in line and "1 row" in line for line in md.report)


def test_rfm_by_hand(tiny):
    res = rfm.build_rfm(_canon(tiny).df).df.set_index("customer_id")
    # Last date in data (after refund removal) = 2025-03-10 -> snapshot 2025-03-11.
    # A: last 2025-03-01 -> 10 days; 3 purchase days; 10+20+30 = 60
    assert res.loc["A", "recency_days"] == 10
    assert res.loc["A", "frequency"] == 3
    assert res.loc["A", "monetary"] == 60
    assert res.loc["A", "tenure_days"] == 59
    # B: two rows on the same day = 1 order (no order id mapped); 75 total
    assert res.loc["B", "recency_days"] == 38
    assert res.loc["B", "frequency"] == 1
    assert res.loc["B", "monetary"] == 75
    assert res.loc["B", "avg_order_value"] == 75
    # C: the bad-date row is gone; last 2025-03-10 -> 1 day
    assert res.loc["C", "recency_days"] == 1
    assert res.loc["C", "frequency"] == 2
    assert res.loc["C", "monetary"] == 20


def test_refunds_excluded_or_kept(tiny):
    canon = _canon(tiny).df
    excl = rfm.build_rfm(canon, exclude_nonpositive=True)
    assert any("refund" in line for line in excl.report)
    kept = rfm.build_rfm(canon, exclude_nonpositive=False).df.set_index("customer_id")
    assert kept.loc["A", "monetary"] == 40      # 60 - 20 refund
    assert kept.loc["A", "recency_days"] == 6   # refund on 03-05, snapshot 03-11


def test_short_range_warning(tiny):
    res = rfm.build_rfm(_canon(tiny).df)
    assert res.warnings == []                     # 68 days >= 30
    short = _canon(tiny.iloc[[3, 4]]).df
    assert rfm.build_rfm(short).warnings
