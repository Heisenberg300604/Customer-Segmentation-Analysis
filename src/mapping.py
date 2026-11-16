"""Column mapper: turn ANY customer/transactions file into the canonical schema.

A "role" is what a column means (amount, date, gender, ...). The mapper:
  1. suggest_mapping()  - guesses which column plays each role, with a
                          confidence level (synonyms, fuzzy match, dtype sniffing);
  2. validate_mapping() - checks the user's final choice;
  3. apply_mapping()    - renames to canonical column names, converts types,
                          reports dropped rows, then reuses the shared pipeline
                          helpers (Yes/No normalising, rating imputation,
                          purchase-rate features).

Canonical column names are the same snake_case names the retail pipeline uses,
so every downstream module (segmentation, fit, dashboard) works on both.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config, pipeline

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
# role -> canonical column name in the mapped DataFrame.
ROLE_TO_COLUMN = {
    "customer_id": "customer_id",
    "order_id": "order_id",
    "transaction_date": "transaction_date",
    "amount": "purchase_amount",
    "frequency": "frequency_of_purchases",
    "previous_purchases": "previous_purchases",
    "discount": "discount_applied",
    "category": "category",
    "item": "item_purchased",
    "gender": "gender",
    "age": "age",
    "rating": "review_rating",
    "subscription": "subscription_status",
    "shipping": "shipping_type",
    "location": "location",
}
ROLES = list(ROLE_TO_COLUMN)
REQUIRED_ROLES = ["amount"]
NUMERIC_ROLES = ["amount", "previous_purchases", "age", "rating"]

ROLE_HELP = {
    "customer_id": "Identifies the customer (needed for RFM).",
    "order_id": "Order / invoice number (optional; improves RFM order counts).",
    "transaction_date": "Purchase date (enables RFM and trend charts).",
    "amount": "Money spent per row. Required.",
    "frequency": "How often the customer buys: a label (Weekly…) or a number per year.",
    "previous_purchases": "Number of earlier purchases.",
    "discount": "Whether a discount was used (Yes/No, 0/1, or a discount amount).",
    "category": "Product category.",
    "item": "Product / item name.",
    "gender": "Customer gender.",
    "age": "Customer age in years.",
    "rating": "Review rating.",
    "subscription": "Subscriber Yes/No.",
    "shipping": "Shipping / delivery type.",
    "location": "City, state or country.",
}

# Synonyms per role, already normalised (lower case, words separated by spaces).
SYNONYMS = {
    "customer_id": ["customer id", "cust id", "customerid", "custid", "user id",
                    "userid", "client id", "client", "member id", "customer",
                    "customer number", "customer no", "account id"],
    "order_id": ["order id", "invoice", "invoice no", "invoice number",
                 "invoiceno", "order number", "order no", "transaction id",
                 "receipt id", "basket id"],
    "transaction_date": ["date", "order date", "invoice date", "purchase date",
                         "transaction date", "timestamp", "invoicedate",
                         "orderdate", "created at", "datetime", "sale date"],
    "amount": ["purchase amount usd", "purchase amount", "amount", "amount usd",
               "order total", "total", "revenue", "sales", "price",
               "total spend", "spend", "order value", "sale amount",
               "total amount", "net amount", "value"],
    "frequency": ["frequency of purchases", "purchase frequency", "frequency",
                  "buying frequency", "purchases per year", "orders per year"],
    "previous_purchases": ["previous purchases", "past purchases",
                           "prior purchases", "previous orders",
                           "number of purchases", "purchase count",
                           "order count", "total orders"],
    "discount": ["discount applied", "discount", "discount used", "discounted",
                 "promo applied", "discount amount", "discount pct",
                 "discount percent", "coupon used", "promo"],
    "category": ["category", "product category", "product group", "product type",
                 "department", "product line", "category name"],
    "item": ["item purchased", "item", "product", "product name", "item name",
             "sku", "description", "stock code"],
    "gender": ["gender", "sex"],
    "age": ["age", "customer age", "age years"],
    "rating": ["review rating", "rating", "stars", "score", "review score",
               "satisfaction"],
    "subscription": ["subscription status", "subscription", "subscriber",
                     "is subscriber", "member", "membership"],
    "shipping": ["shipping type", "shipping", "delivery type", "shipping method",
                 "delivery", "ship mode", "ship method"],
    "location": ["location", "state", "city", "country", "region", "address"],
}

# Role assignment order: specific roles first, so e.g. "Previous Purchases" is
# claimed by previous_purchases before amount looks at it.
_ROLE_ORDER = ["customer_id", "order_id", "transaction_date", "frequency",
               "previous_purchases", "discount", "subscription", "shipping",
               "rating", "amount", "category", "item", "gender", "age", "location"]


def _norm(name: str) -> str:
    """'Purchase Amount (USD)' -> 'purchase amount usd'."""
    return pipeline.to_snake_case(name).replace("_", " ")


def _numeric_share(series: pd.Series) -> float:
    """Share of non-empty values that convert to numbers (after removing $ , %)."""
    s = series.dropna()
    if s.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(s):
        return 1.0
    cleaned = s.astype(str).str.replace(r"[\$,%\s€£₹]", "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _date_share(series: pd.Series, dayfirst: bool = False) -> float:
    """Share of non-empty values that parse as dates (numbers do not count)."""
    s = series.dropna()
    if s.empty or pd.api.types.is_numeric_dtype(s):
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(s):
        return 1.0
    sample = s.astype(str).head(500)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed",
                            dayfirst=dayfirst)
    return float(parsed.notna().mean())


def _dtype_ok(role: str, series: pd.Series) -> bool:
    if role in NUMERIC_ROLES:
        return _numeric_share(series) >= 0.9
    if role == "transaction_date":
        return _date_share(series) >= 0.8
    return True


# ---------------------------------------------------------------------------
# 1. Suggest
# ---------------------------------------------------------------------------
@dataclass
class Suggestion:
    column: str | None
    confidence: str          # "high" | "medium" | "low" | "none"
    reason: str


def suggest_mapping(df: pd.DataFrame, with_confidence: bool = False):
    """Guess the column for every role.

    Matching, best first: exact synonym (high) -> all synonym words present in
    the column name (medium) -> fuzzy match with difflib (low) -> dtype
    sniffing for dates and gender (low). Candidates whose values do not fit
    the role (e.g. text for 'amount') are skipped. Each column is used once.

    Returns {role: column or None}, or {role: Suggestion} if with_confidence.
    """
    columns = list(df.columns)
    normed = {c: _norm(c) for c in columns}
    used: set[str] = set()
    result: dict[str, Suggestion] = {}

    for role in _ROLE_ORDER:
        syns = SYNONYMS[role]
        best: Suggestion | None = None
        free = [c for c in columns if c not in used]

        # (1) exact synonym
        for c in free:
            if normed[c] in syns and _dtype_ok(role, df[c]):
                best = Suggestion(c, "high", "name matches a known synonym")
                break
        # (2) every word of a synonym appears in the column name
        if best is None:
            scored = []
            for c in free:
                words = set(normed[c].split())
                for rank, syn in enumerate(syns):
                    if set(syn.split()) <= words and _dtype_ok(role, df[c]):
                        scored.append((rank, len(words), c, syn))
                        break
            if scored:
                _, _, c, syn = min(scored)
                best = Suggestion(c, "medium", f"name contains '{syn}'")
        # (3) fuzzy match on the whole name
        if best is None:
            by_norm = {normed[c]: c for c in free}
            for syn in syns:
                hit = difflib.get_close_matches(syn, list(by_norm), n=1, cutoff=0.8)
                if hit and _dtype_ok(role, df[by_norm[hit[0]]]):
                    best = Suggestion(by_norm[hit[0]], "low",
                                      f"name looks like '{syn}'")
                    break
        # (4) dtype sniffing
        if best is None and role == "transaction_date":
            for c in free:
                if not pd.api.types.is_numeric_dtype(df[c]) and _date_share(df[c]) >= 0.9:
                    best = Suggestion(c, "low", "values look like dates")
                    break
        if best is None and role == "gender":
            for c in free:
                vals = set(df[c].dropna().astype(str).str.strip().str.lower().unique())
                if vals and vals <= {"m", "f", "male", "female", "other", "x"}:
                    best = Suggestion(c, "low", "values look like gender")
                    break

        if best is None:
            best = Suggestion(None, "none", "no matching column found")
        else:
            used.add(best.column)
        result[role] = best

    ordered = {r: result[r] for r in ROLES}
    if with_confidence:
        return ordered
    return {r: s.column for r, s in ordered.items()}


# ---------------------------------------------------------------------------
# 2. Validate
# ---------------------------------------------------------------------------
def validate_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> list[str]:
    """Return a list of problems with a mapping (empty list = OK)."""
    errors = []
    for role in REQUIRED_ROLES:
        if not mapping.get(role):
            errors.append(f"'{role}' is required - choose the column that holds it.")
    chosen = [(r, c) for r, c in mapping.items() if c]
    seen: dict[str, str] = {}
    for role, col in chosen:
        if col not in df.columns:
            errors.append(f"'{role}': column '{col}' is not in the file.")
            continue
        if col in seen:
            errors.append(f"Column '{col}' is used for both '{seen[col]}' and "
                          f"'{role}'. Each column can play one role.")
        seen[col] = role
        if role in NUMERIC_ROLES and _numeric_share(df[col]) < 0.5:
            errors.append(f"'{role}' must be numeric, but most values in "
                          f"'{col}' are not numbers.")
        if role == "transaction_date" and _date_share(df[col]) < 0.5:
            errors.append(f"'transaction_date': most values in '{col}' are "
                          "not dates.")
    return errors


# ---------------------------------------------------------------------------
# 3. Value-level helpers
# ---------------------------------------------------------------------------
def frequency_is_numeric(series: pd.Series) -> bool:
    """True when the frequency column already holds numbers (per year)."""
    return _numeric_share(series) >= 0.9


def default_frequency_map(series: pd.Series) -> dict[str, float | None]:
    """Unique frequency labels -> purchases/year, pre-filled where known.

    Unknown labels map to None: the user must fill them in before continuing.
    """
    known = {k.lower(): v for k, v in config.DEFAULT_FREQUENCY_LABELS.items()}
    labels = sorted(series.dropna().astype(str).str.strip().unique())
    return {lab: known.get(lab.lower()) for lab in labels}


def normalise_discount(series: pd.Series) -> tuple[pd.Series, str]:
    """Turn any discount column into a 0/1 flag and explain the interpretation.

    Handles Yes/No and True/False text, 0/1 values, and numeric discount
    amounts or percentages (any value > 0 counts as 'discounted').
    """
    text = series.astype(str).str.strip().str.lower()
    yes_no = pipeline._YES_VALUES | pipeline._NO_VALUES | {"nan", "none", ""}
    if text.isin(yes_no).all():
        flag = text.isin(pipeline._YES_VALUES).astype(int)
        values = set(text.unique()) - {"nan", "none", ""}
        if values <= {"0", "1", "0.0", "1.0"}:
            how = "0/1 values: 1 = discounted."
        elif values & {"true", "false", "t", "f"}:
            how = "True/False values: True = discounted."
        else:
            how = "Yes/No values: Yes = discounted."
        return flag, how
    numbers = pd.to_numeric(
        series.astype(str).str.replace(r"[\$,%\s€£₹]", "", regex=True),
        errors="coerce")
    if numbers.notna().mean() >= 0.9:
        flag = (numbers.fillna(0) > 0).astype(int)
        return flag, ("Numeric discount amounts/percentages: any value above 0 "
                      "counts as discounted.")
    bad = sorted(set(series.dropna().astype(str)) - set(series[numbers.notna()]
                                                        .astype(str)))[:5]
    raise ValueError(f"Could not read the discount column. Unexpected values: {bad}")


def _to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,%\s€£₹]", "", regex=True),
        errors="coerce")


# ---------------------------------------------------------------------------
# 4. Apply
# ---------------------------------------------------------------------------
@dataclass
class MappedData:
    """Result of apply_mapping()."""
    df: pd.DataFrame
    schema: dict[str, str]                 # role -> canonical column present
    report: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    value_maps: dict | None = None,
    dayfirst: bool = False,
) -> MappedData:
    """Rename/convert a file into the canonical schema.

    Args:
        df: the uploaded file.
        mapping: {role: column or None}.
        value_maps: {"frequency": {label: per_year}} for text frequency columns.
        dayfirst: parse ambiguous dates like 03/04/2024 as 3 April.

    Raises ValueError for an invalid mapping or unfilled frequency labels.
    """
    errors = validate_mapping(df, mapping)
    if errors:
        raise ValueError(" ".join(errors))
    value_maps = value_maps or {}
    report: list[str] = []

    out = pd.DataFrame(index=df.index)
    for role, col in mapping.items():
        if col:
            out[ROLE_TO_COLUMN[role]] = df[col]
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    rows_in = len(out)

    # Amount: must be numeric; drop rows that are not.
    out["purchase_amount"] = _to_number(out["purchase_amount"])
    bad = out["purchase_amount"].isna()
    if bad.any():
        report.append(f"Dropped {int(bad.sum())} row(s) with a missing or "
                      "non-numeric amount.")
        out = out[~bad]

    # Dates.
    if "transaction_date" in out.columns:
        parsed = pd.to_datetime(out["transaction_date"], errors="coerce",
                                format="mixed", dayfirst=dayfirst)
        bad = parsed.isna()
        if bad.any():
            report.append(f"Dropped {int(bad.sum())} row(s) with an unreadable date.")
        out["transaction_date"] = parsed
        out = out[~bad]

    for col in ["previous_purchases", "age", "review_rating"]:
        if col in out.columns:
            out[col] = _to_number(out[col])

    # Frequency -> purchases per year.
    if "frequency_of_purchases" in out.columns:
        freq = out["frequency_of_purchases"]
        if frequency_is_numeric(freq):
            out["purchase_rate_per_year"] = _to_number(freq)
            report.append("Frequency column is numeric: used as purchases per year.")
        else:
            fmap = value_maps.get("frequency") or default_frequency_map(freq)
            fmap = {str(k).strip().lower(): v for k, v in fmap.items()}
            labels = freq.dropna().astype(str).str.strip()
            missing = sorted({lab for lab in labels.unique()
                              if fmap.get(lab.lower()) in (None, "")
                              or pd.isna(fmap.get(lab.lower()))})
            if missing:
                raise ValueError(
                    "Give a 'purchases per year' number for every frequency "
                    f"label. Missing: {missing}")
            out["purchase_rate_per_year"] = (
                freq.astype(str).str.strip().str.lower().map(fmap).astype(float))
        n_bad = int(out["purchase_rate_per_year"].isna().sum())
        if n_bad:
            report.append(f"{n_bad} row(s) have no frequency value; they are "
                          "kept but cannot be segmented on frequency.")

    # Discount -> 0/1 flag (+ Yes/No text for display).
    if "discount_applied" in out.columns:
        flag, how = normalise_discount(out["discount_applied"])
        out["discount_flag"] = flag
        out["discount_applied"] = np.where(flag == 1, "Yes", "No")
        report.append(f"Discount interpreted as {how}")

    if "subscription_status" in out.columns:
        out["subscription_status"] = pipeline.normalise_yes_no(
            out["subscription_status"], "subscription_status", strict=False)

    # Shared pipeline steps: rating imputation and frequency-days feature.
    if "review_rating" in out.columns:
        n_missing = int(out["review_rating"].isna().sum())
        if n_missing and out["review_rating"].notna().any():
            out["review_rating"] = pipeline.impute_rating(out, None, None)
            report.append(f"Filled {n_missing} missing rating(s) with the "
                          "category (or overall) median.")
    # Age groups for a foreign file are its own quartiles (see pipeline), so
    # age is set aside while the shared engineer_features() runs.
    age = out.pop("age") if "age" in out.columns else None
    out = pipeline.engineer_features(out)
    if age is not None:
        out["age"] = age
        out["age_group"] = pipeline.age_quartile_groups(age)

    out = out.reset_index(drop=True)
    schema = {role: ROLE_TO_COLUMN[role] for role, col in mapping.items() if col}
    report.insert(0, f"{rows_in:,} rows read, {len(out):,} kept.")
    return MappedData(df=out, schema=schema, report=report,
                      rows_in=rows_in, rows_out=len(out))


# ---------------------------------------------------------------------------
# 5. Save / load a mapping
# ---------------------------------------------------------------------------
def mapping_to_json(mapping: dict, value_maps: dict | None = None,
                    dayfirst: bool = False) -> str:
    return json.dumps({"mapping": mapping, "value_maps": value_maps or {},
                       "dayfirst": dayfirst}, indent=2)


def mapping_from_json(text: str) -> tuple[dict, dict, bool]:
    data = json.loads(text)
    mapping = {r: data.get("mapping", {}).get(r) for r in ROLES}
    return mapping, data.get("value_maps", {}), bool(data.get("dayfirst", False))
