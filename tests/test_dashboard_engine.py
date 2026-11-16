"""Tests for src/dashboard_engine.py on three very different schemas."""
import pandas as pd
import pytest

from src import config, dashboard_engine as de, mapping


@pytest.fixture(scope="module")
def retail():
    if not config.CLUSTERED_CSV_PATH.exists():
        pytest.skip("Run `python -m src.train` first")
    return pd.read_csv(config.CLUSTERED_CSV_PATH)


def test_full_retail(retail):
    d = de.build_dashboard(retail, de.RETAIL_SCHEMA)
    keys = d.chart_keys()
    for k in ["subscription_donut", "revenue_by_category", "sales_by_category",
              "revenue_by_age_group", "sales_by_age_group", "segment_donut",
              "revenue_by_segment", "segment_x_category", "revenue_by_gender",
              "top_rated_products", "discount_dependent_products",
              "avg_spend_by_shipping"]:
        assert k in keys, k
    labels = dict(d.kpis)
    assert labels["Customers"] == "3,900"
    assert "monthly_trend" not in keys  # no dates in the retail file


def test_minimal_amount_only():
    df = pd.DataFrame({"purchase_amount": [10.0, 20.0, 30.0]})
    d = de.build_dashboard(df, {"amount": "purchase_amount"})
    assert d.charts == []
    assert dict(d.kpis)["Total Revenue"] == "$60.00"
    assert any("Category" in n for n in d.notes)


def test_transactions_with_dates():
    path = config.DATA_DIR / "sample_transactions.csv"
    if not path.exists():
        pytest.skip("synthetic file missing")
    tx = pd.read_csv(path)
    md = mapping.apply_mapping(tx, mapping.suggest_mapping(tx))
    d = de.build_dashboard(md.df, md.schema)
    assert "monthly_trend" in d.chart_keys()
    assert "subscription_donut" not in d.chart_keys()


def test_filters_and_empty(retail):
    cols = de.filter_columns(retail, de.RETAIL_SCHEMA)
    assert "gender" in cols and "segment" in cols
    only_f = de.apply_filters(retail, {"gender": ["Female"]})
    assert set(only_f["gender"]) == {"Female"}
    empty = de.build_dashboard(retail.iloc[0:0], de.RETAIL_SCHEMA)
    assert empty.notes and not empty.charts
