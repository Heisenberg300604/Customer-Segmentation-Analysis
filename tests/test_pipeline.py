"""Tests for src/pipeline.py.

Two groups:
  * Synthetic tests: run on a tiny hand-made DataFrame, always run.
  * Real-data tests: check the cleaned Customer_dataset.csv reproduces the
    verified baseline numbers. Skipped if data/Customer_dataset.csv is absent.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    """Eight raw rows with the real column names and a few messy values."""
    return pd.DataFrame({
        "Customer ID": range(1, 9),
        "Age": [19, 25, 33, 41, 47, 55, 63, 70],
        "Gender": ["Male", "Female"] * 4,
        "Item Purchased": ["Shirt"] * 8,
        "Category": ["Clothing", "Clothing", "Footwear", "Footwear",
                     "Clothing", "Accessories", "Footwear", "Clothing"],
        "Purchase Amount (USD)": [20, 40, 60, 80, 30, 50, 70, 90],
        "Review Rating": [3.0, np.nan, 4.0, 5.0, 4.0, np.nan, np.nan, 2.0],
        "Subscription Status": ["Yes", "No", " yes ", "NO", "No", "No", "Yes", "No"],
        "Shipping Type": ["Express", "Standard"] * 4,
        "Discount Applied": ["Yes", "No", "yes", "No", "No", "Yes", "No", "No"],
        "Promo Code Used": ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No"],
        "Previous Purchases": [1, 5, 10, 20, 30, 40, 45, 50],
        "Frequency of Purchases": ["Weekly", "Bi-Weekly", "Fortnightly",
                                   "Monthly", "Quarterly", "Every 3 Months",
                                   "Annually", " weekly "],
    })


@pytest.fixture(scope="module")
def real_clean() -> pd.DataFrame:
    """The real dataset, cleaned and feature-engineered."""
    if not config.RAW_DATA_PATH.exists():
        pytest.skip(f"{config.RAW_DATA_PATH} not found - add the dataset to run")
    raw = pipeline.load_raw()
    return pipeline.engineer_features(pipeline.clean(raw))


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------
def test_snake_case():
    assert pipeline.to_snake_case("Purchase Amount (USD)") == "purchase_amount_usd"
    assert pipeline.to_snake_case("﻿Customer ID ") == "customer_id"


def test_validate_columns_all_present(tiny_raw):
    ok, missing_model, missing_dash = pipeline.validate_columns(tiny_raw)
    assert ok and missing_model == []
    assert missing_dash == ["Location"]  # tiny_raw has no Location column


def test_validate_columns_reports_missing(tiny_raw):
    ok, missing_model, _ = pipeline.validate_columns(
        tiny_raw.drop(columns=["Previous Purchases", "Gender"])
    )
    assert not ok
    assert missing_model == ["Previous Purchases"]


def test_clean_missing_model_column_raises(tiny_raw):
    with pytest.raises(ValueError, match="Discount Applied"):
        pipeline.clean(tiny_raw.drop(columns=["Discount Applied"]))


def test_clean_unknown_frequency_raises(tiny_raw):
    bad = tiny_raw.copy()
    bad.loc[0, "Frequency of Purchases"] = "Daily"
    with pytest.raises(ValueError, match="Daily"):
        pipeline.clean(bad)


def test_clean_bad_discount_value_raises(tiny_raw):
    bad = tiny_raw.copy()
    bad.loc[0, "Discount Applied"] = "maybe"
    with pytest.raises(ValueError, match="maybe"):
        pipeline.clean(bad)


def test_clean_non_numeric_amount_raises(tiny_raw):
    bad = tiny_raw.copy().astype({"Purchase Amount (USD)": object})
    bad.loc[0, "Purchase Amount (USD)"] = "abc"
    with pytest.raises(ValueError, match="purchase_amount"):
        pipeline.clean(bad)


def test_clean_basic_properties(tiny_raw):
    out = pipeline.clean(tiny_raw)
    assert len(out) == len(tiny_raw)
    assert "promo_code_used" not in out.columns
    assert "purchase_amount" in out.columns
    assert out["review_rating"].isna().sum() == 0
    assert set(out["discount_applied"]) == {"Yes", "No"}
    assert set(out["subscription_status"]) == {"Yes", "No"}
    assert out.loc[7, "frequency_of_purchases"] == "Weekly"  # ' weekly ' fixed
    # Input must not be modified.
    assert tiny_raw["Review Rating"].isna().sum() == 3


def test_rating_imputed_by_category_median(tiny_raw):
    out = pipeline.clean(tiny_raw)
    # Clothing ratings present: 3.0, 4.0, 2.0 -> median 3.0 (row 1 is Clothing)
    assert out.loc[1, "review_rating"] == 3.0
    # Footwear ratings present: 4.0, 5.0 -> median 4.5 (row 6 is Footwear)
    assert out.loc[6, "review_rating"] == 4.5
    # Accessories has no rating at all -> global median of 3,4,5,4,2 = 4.0
    assert out.loc[5, "review_rating"] == 4.0


def test_rating_uses_stored_medians_and_fallback(tiny_raw):
    out = pipeline.clean(tiny_raw, rating_medians={"Clothing": 1.5},
                         global_rating_median=2.5)
    assert out.loc[1, "review_rating"] == 1.5   # Clothing -> stored median
    assert out.loc[6, "review_rating"] == 2.5   # Footwear unseen -> fallback


def test_engineer_features(tiny_raw):
    out = pipeline.engineer_features(pipeline.clean(tiny_raw))
    assert out["purchase_rate_per_year"].tolist() == [
        52, config.BI_WEEKLY_PER_YEAR, 26, 12, 4, 4, 1, 52]
    assert out.loc[0, "purchase_frequency_days"] == 7      # 365/52
    assert out.loc[6, "purchase_frequency_days"] == 365    # annually
    assert out["discount_flag"].tolist() == [1, 0, 1, 0, 0, 1, 0, 0]
    assert set(out["age_group"]) == set(config.AGE_GROUP_LABELS)


def test_stored_age_edges_match_qcut(tiny_raw):
    """pd.cut on stored edges must give the same groups as the original qcut."""
    cleaned = pipeline.clean(tiny_raw)
    edges = pipeline.compute_age_bin_edges(cleaned["age"])
    by_qcut = pipeline.engineer_features(cleaned)["age_group"]
    by_edges = pipeline.engineer_features(cleaned, age_bin_edges=edges)["age_group"]
    assert by_qcut.tolist() == by_edges.tolist()


def test_stored_age_edges_handle_out_of_range(tiny_raw):
    cleaned = pipeline.clean(tiny_raw)
    edges = pipeline.compute_age_bin_edges(cleaned["age"])
    cleaned.loc[0, "age"] = 5     # younger than anyone in training
    cleaned.loc[7, "age"] = 99    # older than anyone in training
    out = pipeline.engineer_features(cleaned, age_bin_edges=edges)
    assert out.loc[0, "age_group"] == "Young Adult"
    assert out.loc[7, "age_group"] == "Senior"


def test_build_features_order(tiny_raw):
    df = pipeline.engineer_features(pipeline.clean(tiny_raw))
    X = pipeline.build_features(df)
    assert list(X.columns) == config.MODEL_FEATURES
    assert X.dtypes.eq(float).all()


def test_build_features_requires_engineering(tiny_raw):
    with pytest.raises(ValueError, match="engineer_features"):
        pipeline.build_features(pipeline.clean(tiny_raw))


# ---------------------------------------------------------------------------
# Real-data tests (verified baseline numbers from the SQL / Power BI project)
# ---------------------------------------------------------------------------
def test_real_shape_and_nulls(real_clean):
    assert len(real_clean) == 3900
    assert real_clean["review_rating"].isna().sum() == 0
    assert "promo_code_used" not in real_clean.columns


def test_real_seven_frequency_labels(real_clean):
    assert set(real_clean["frequency_of_purchases"]) == set(
        config.FREQUENCY_TO_PER_YEAR)


def test_real_revenue_by_gender(real_clean):
    revenue = real_clean.groupby("gender")["purchase_amount"].sum()
    assert revenue["Female"] == 75191
    assert revenue["Male"] == 157890


def test_real_revenue_by_age_group(real_clean):
    revenue = real_clean.groupby("age_group")["purchase_amount"].sum()
    assert revenue["Young Adult"] == 62143
    assert revenue["Middle-aged"] == 59197
    assert revenue["Adult"] == 55978
    assert revenue["Senior"] == 55763


def test_real_means(real_clean):
    assert round(real_clean["purchase_amount"].mean(), 2) == 59.76
    assert round(real_clean["review_rating"].mean(), 2) == 3.75
