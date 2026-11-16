"""Tests for src/mapping.py (column mapper)."""
import pandas as pd
import pytest

from src import config, mapping, pipeline


@pytest.fixture(scope="module")
def raw():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("dataset missing")
    return pipeline.load_raw()


RENAMES = {"Customer ID": "cust_id", "Purchase Amount (USD)": "order total",
           "Frequency of Purchases": "purchase_frequency",
           "Previous Purchases": "Past Purchases", "Discount Applied": "discount",
           "Category": "product group", "Gender": "sex", "Age": "customer_age",
           "Review Rating": "stars", "Subscription Status": "subscriber",
           "Shipping Type": "delivery type", "Item Purchased": "product name",
           "Location": "state"}


def test_retail_default_mapping(raw):
    m = mapping.suggest_mapping(raw)
    assert m["amount"] == "Purchase Amount (USD)"
    assert m["frequency"] == "Frequency of Purchases"
    assert m["previous_purchases"] == "Previous Purchases"
    assert m["discount"] == "Discount Applied"
    assert m["transaction_date"] is None


def test_retail_through_mapper_matches_pipeline(raw):
    md = mapping.apply_mapping(raw, mapping.suggest_mapping(raw))
    ref = pipeline.engineer_features(pipeline.clean(raw))
    assert md.rows_out == len(ref)
    for col in ["purchase_rate_per_year", "purchase_amount", "discount_flag",
                "previous_purchases"]:
        assert (md.df[col].to_numpy() == ref[col].to_numpy()).all(), col
    assert md.df["review_rating"].isna().sum() == 0


def test_renamed_columns_auto_mapped(raw):
    renamed = raw.rename(columns=RENAMES)
    m = mapping.suggest_mapping(renamed)
    expected = {"customer_id": "cust_id", "amount": "order total",
                "frequency": "purchase_frequency",
                "previous_purchases": "Past Purchases", "discount": "discount",
                "category": "product group", "gender": "sex",
                "age": "customer_age", "rating": "stars",
                "subscription": "subscriber", "shipping": "delivery type",
                "item": "product name", "location": "state"}
    for role, col in expected.items():
        assert m[role] == col, (role, m[role])


def test_rejects_duplicate_and_non_numeric(raw):
    m = mapping.suggest_mapping(raw)
    dup = {**m, "age": m["amount"]}
    assert any("used for both" in e for e in mapping.validate_mapping(raw, dup))
    bad = {**m, "amount": "Gender"}
    assert any("must be numeric" in e for e in mapping.validate_mapping(raw, bad))
    with pytest.raises(ValueError):
        mapping.apply_mapping(raw, bad)
    missing = {**m, "amount": None}
    assert any("required" in e for e in mapping.validate_mapping(raw, missing))


def test_unknown_frequency_labels_block_until_filled():
    df = pd.DataFrame({"amt": [10, 20, 30], "freq": ["Weekly", "Sometimes", "Daily"]})
    m = {r: None for r in mapping.ROLES} | {"amount": "amt", "frequency": "freq"}
    defaults = mapping.default_frequency_map(df["freq"])
    assert defaults["Sometimes"] is None and defaults["Weekly"] == 52
    with pytest.raises(ValueError, match="Sometimes"):
        mapping.apply_mapping(df, m)
    filled = {**defaults, "Sometimes": 3}
    md = mapping.apply_mapping(df, m, value_maps={"frequency": filled})
    assert md.df["purchase_rate_per_year"].tolist() == [52, 3, 365]


@pytest.mark.parametrize("values,expected", [
    (["Yes", "No", "yes"], [1, 0, 1]),
    (["True", "False", "TRUE"], [1, 0, 1]),
    ([1, 0, 1], [1, 0, 1]),
    (["5%", "0%", "12.5%"], [1, 0, 1]),
    ([0.0, 3.5, 10.0], [0, 1, 1]),
])
def test_discount_normalisation(values, expected):
    flag, how = mapping.normalise_discount(pd.Series(values))
    assert flag.tolist() == expected
    assert how


def test_mapping_json_roundtrip():
    m = {r: None for r in mapping.ROLES} | {"amount": "x"}
    text = mapping.mapping_to_json(m, {"frequency": {"A": 1}}, dayfirst=True)
    m2, vm, dayfirst = mapping.mapping_from_json(text)
    assert m2 == m and vm == {"frequency": {"A": 1}} and dayfirst
