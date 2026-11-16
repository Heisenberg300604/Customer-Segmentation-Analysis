"""The shared cleaning + feature-engineering pipeline.

This is the ONE place where a "clean row" is defined. Both `src/train.py` and
the Streamlit app import these functions, so an uploaded file is transformed
exactly the same way as the training data was.

Typical use:
    df = clean(raw_df)
    df = engineer_features(df)
    X = build_features(df)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def to_snake_case(name: str) -> str:
    """Turn a column name like 'Purchase Amount (USD)' into 'purchase_amount_usd'.

    Steps: strip spaces (and a stray BOM), lower-case, replace every run of
    non-letters/digits with one underscore, trim leading/trailing underscores.
    """
    name = str(name).replace("﻿", "").strip().lower()
    name = re.sub(r"[^0-9a-z]+", "_", name)
    return name.strip("_")


def _canonical(name: str) -> str:
    """snake_case a column name and apply the short aliases from config."""
    snake = to_snake_case(name)
    return config.COLUMN_ALIASES.get(snake, snake)


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df whose columns are canonical snake_case names."""
    out = df.copy()
    out.columns = [_canonical(c) for c in out.columns]
    return out


# Text values that we accept as "Yes" / "No" (compared in lower case).
_YES_VALUES = {"yes", "y", "true", "t", "1", "1.0"}
_NO_VALUES = {"no", "n", "false", "f", "0", "0.0"}


def normalise_yes_no(series: pd.Series, column: str,
                      strict: bool) -> pd.Series:
    """Map yes/no-like values to exactly 'Yes' or 'No'.

    strict=True (model columns): raise ValueError listing any value that is not
    recognisable, so a typo is reported instead of silently becoming 'No'.
    strict=False (dashboard-only columns): unrecognised values become NaN.
    """
    text = series.astype(str).str.strip().str.lower()
    result = pd.Series(
        np.select([text.isin(_YES_VALUES), text.isin(_NO_VALUES)],
                  ["Yes", "No"], default=None),
        index=series.index, dtype=object,
    )
    unknown = sorted(set(text[result.isna()]))
    if unknown and strict:
        raise ValueError(
            f"Column '{column}' must contain Yes/No values. "
            f"Unrecognised values: {unknown}"
        )
    return result


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    """Check which required columns a raw DataFrame has.

    Column names are compared after snake_casing, so 'Purchase Amount (USD)',
    'purchase_amount_usd' and ' purchase amount (usd) ' all count as present.

    Returns:
        ok: True when every model-required column is present.
        missing_model: model columns that are missing (blocks segmentation).
        missing_dashboard: dashboard columns that are missing (charts hidden).
    """
    present = {_canonical(c) for c in df.columns}
    missing_model = [c for c in config.REQUIRED_COLUMNS_MODEL
                     if _canonical(c) not in present]
    missing_dashboard = [c for c in config.REQUIRED_COLUMNS_DASHBOARD
                         if _canonical(c) not in present]
    return len(missing_model) == 0, missing_model, missing_dashboard


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def clean(
    df: pd.DataFrame,
    rating_medians: dict[str, float] | None = None,
    global_rating_median: float | None = None,
) -> pd.DataFrame:
    """Clean a raw transactions DataFrame.

    What it does, in order:
      1. Renames columns to snake_case (e.g. 'Purchase Amount (USD)' ->
         'purchase_amount') and checks the model columns are present.
      2. Strips extra spaces from every text value.
      3. Normalises Yes/No columns to exactly 'Yes'/'No'.
      4. Standardises frequency labels and raises an error on unknown labels.
      5. Converts numeric columns to numbers and refuses missing/invalid values
         in the model columns.
      6. Fills missing review ratings with the median rating of the same
         category (falls back to the overall median for unseen categories).
      7. Drops 'promo_code_used' (it is identical to 'discount_applied').

    Args:
        df: the raw DataFrame (raw or snake_case column names both work).
        rating_medians: per-category medians to use for imputation. Training
            passes None (medians are computed from the data itself); the app
            passes the training medians stored in the model artifact.
        global_rating_median: overall median used when a category is unseen.

    Returns:
        A new, cleaned DataFrame. The input is not modified.
    """
    ok, missing_model, _ = validate_columns(df)
    if not ok:
        raise ValueError(
            "Missing required column(s) for segmentation: "
            + ", ".join(missing_model)
        )

    out = normalise_column_names(df)

    # 2. Strip whitespace in every text column.
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip()

    # 3. Yes/No columns.
    for col in config.YES_NO_COLUMNS:
        if col in out.columns:
            out[col] = normalise_yes_no(
                out[col], col, strict=(col == "discount_applied")
            )

    # 4. Frequency labels: match case-insensitively to the known labels.
    known = {label.lower(): label for label in config.FREQUENCY_TO_PER_YEAR}
    freq_lower = out["frequency_of_purchases"].astype(str).str.strip().str.lower()
    unknown = sorted(set(out.loc[~freq_lower.isin(known), "frequency_of_purchases"]
                         .astype(str)))
    if unknown:
        raise ValueError(
            f"Unknown 'Frequency of Purchases' label(s): {unknown}. "
            f"Allowed labels: {list(config.FREQUENCY_TO_PER_YEAR)}"
        )
    out["frequency_of_purchases"] = freq_lower.map(known)

    # 5. Numeric columns. Model columns must be complete and numeric.
    for col in ["purchase_amount", "previous_purchases"]:
        converted = pd.to_numeric(out[col], errors="coerce")
        n_bad = int(converted.isna().sum())
        if n_bad:
            raise ValueError(
                f"Column '{col}' has {n_bad} missing or non-numeric value(s)."
            )
        out[col] = converted
    for col in ["age", "review_rating"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # 6. Impute review ratings by category median.
    if "review_rating" in out.columns:
        out["review_rating"] = impute_rating(
            out, rating_medians, global_rating_median
        )

    # 7. Drop redundant columns.
    out = out.drop(columns=[c for c in config.REDUNDANT_COLUMNS
                            if c in out.columns])
    return out


def rating_medians_by_category(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Per-category and overall median review rating of a snake_case DataFrame.

    Used by training to store the medians in the model artifact, so uploads
    are imputed with the *training* medians.
    """
    global_median = float(df["review_rating"].median())
    if "category" not in df.columns:
        return {}, global_median
    medians = df.groupby("category")["review_rating"].median().dropna()
    return {str(k): float(v) for k, v in medians.items()}, global_median


def impute_rating(
    df: pd.DataFrame,
    rating_medians: dict[str, float] | None,
    global_rating_median: float | None,
) -> pd.Series:
    """Fill missing review ratings with the category median (see clean())."""
    rating = df["review_rating"]
    if rating_medians is None:
        rating_medians, computed_global = rating_medians_by_category(df)
        if global_rating_median is None:
            global_rating_median = computed_global
    if global_rating_median is None:
        global_rating_median = float(rating.median())

    if "category" in df.columns:
        fill = df["category"].map(rating_medians)
    else:
        fill = pd.Series(np.nan, index=df.index)
    fill = fill.fillna(global_rating_median)
    return rating.fillna(fill)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def compute_age_bin_edges(age: pd.Series) -> list[float]:
    """Quartile edges of the age column (5 numbers: min, Q1, Q2, Q3, max).

    Training stores these in the model artifact so uploads reuse the training
    age groups instead of computing new quartiles.
    """
    _, edges = pd.qcut(age, 4, retbins=True)
    return [float(e) for e in edges]


def engineer_features(
    df: pd.DataFrame, age_bin_edges: list[float] | None = None
) -> pd.DataFrame:
    """Add the engineered columns to a cleaned DataFrame.

    New columns:
      - purchase_rate_per_year: purchases per year from the frequency label.
      - purchase_frequency_days: average days between purchases (365 / rate).
      - discount_flag: 1 if a discount was applied, else 0.
      - age_group: Young Adult / Adult / Middle-aged / Senior (only if 'age'
        exists). With age_bin_edges=None (training) it uses the age quartiles
        of this data via pd.qcut. With stored edges (uploads) it uses pd.cut on
        those training edges, opening the outer edges so that ages outside the
        training range still get the youngest/oldest group.
    """
    out = df.copy()

    # Purchases per year. Mapped (foreign) data may already carry this column,
    # computed from the user's own label -> number table; keep it if so.
    if ("purchase_rate_per_year" not in out.columns
            and "frequency_of_purchases" in out.columns):
        out["purchase_rate_per_year"] = (
            out["frequency_of_purchases"].map(config.FREQUENCY_TO_PER_YEAR)
            .astype(float))
    if "purchase_rate_per_year" in out.columns:
        rate = out["purchase_rate_per_year"].where(out["purchase_rate_per_year"] > 0)
        days = (365 / rate).round()
        out["purchase_frequency_days"] = (
            days.astype(int) if days.notna().all() else days)

    if "discount_flag" not in out.columns and "discount_applied" in out.columns:
        out["discount_flag"] = (out["discount_applied"] == "Yes").astype(int)

    if "age" in out.columns and out["age"].notna().any():
        if age_bin_edges is None:
            out["age_group"] = pd.qcut(
                out["age"], 4, labels=config.AGE_GROUP_LABELS
            )
        else:
            edges = [-np.inf, *age_bin_edges[1:-1], np.inf]
            out["age_group"] = pd.cut(
                out["age"], bins=edges, labels=config.AGE_GROUP_LABELS
            )
        # Plain strings are simpler to filter and save than a Categorical.
        out["age_group"] = out["age_group"].astype(str).replace("nan", np.nan)
    return out


def build_features(
    df: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Return the model's feature matrix with columns in a fixed order.

    Args:
        df: cleaned + engineered DataFrame.
        features: column names in the order the model expects. Defaults to
            config.MODEL_FEATURES; the trained model stores its own list.
    """
    features = list(features or config.MODEL_FEATURES)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature column(s) missing: {missing}. "
            "Run clean() and engineer_features() first."
        )
    return df[features].astype(float)


def prepare(
    df_raw: pd.DataFrame,
    rating_medians: dict[str, float] | None = None,
    global_rating_median: float | None = None,
    age_bin_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: clean() then engineer_features()."""
    cleaned = clean(df_raw, rating_medians, global_rating_median)
    return engineer_features(cleaned, age_bin_edges)


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read a raw CSV, tolerating a UTF-8 byte-order mark."""
    return pd.read_csv(path, encoding="utf-8-sig")


def age_quartile_groups(age: pd.Series) -> pd.Series:
    """Age groups for a *foreign* file: its own age quartiles, labelled with
    their ranges (e.g. 'Q1 18–30'), because the retail training edges and
    names do not apply to other customer bases."""
    age = pd.to_numeric(age, errors="coerce")
    if age.notna().sum() < 4:
        return pd.Series(np.nan, index=age.index, dtype=object)
    groups, edges = pd.qcut(age, 4, retbins=True, duplicates="drop")
    labels = [f"Q{i + 1} {edges[i]:.0f}–{edges[i + 1]:.0f}"
              for i in range(len(edges) - 1)]
    out = pd.qcut(age, 4, labels=labels, duplicates="drop")
    return out.astype(str).replace("nan", np.nan)
