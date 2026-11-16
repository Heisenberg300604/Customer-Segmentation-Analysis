"""Reusable K-Means building blocks: k-sweep, k selection, auto-labelling,
segment profiles and prediction with a frozen model.

`train.py` (the retail model) and `fit.py` ("Fit on my data" in the app) both
use these functions, so the method is identical for any dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src import config, pipeline


# ---------------------------------------------------------------------------
# 1. k-sweep
# ---------------------------------------------------------------------------
def kmeans_sweep(
    X_scaled: np.ndarray,
    k_values,
    silhouette_sample: int | None = None,
) -> pd.DataFrame:
    """Fit K-Means for every k and record inertia, silhouette and cluster sizes.

    Args:
        X_scaled: standardised feature matrix.
        k_values: iterable of k to try.
        silhouette_sample: if set, compute silhouette on a fixed-seed random
            sample of at most this many rows (faster on big files).

    Returns:
        DataFrame with one row per k: k, inertia, silhouette, min_cluster_share.
    """
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT,
                    random_state=config.RANDOM_STATE).fit(X_scaled)
        sil = silhouette_score(
            X_scaled, km.labels_,
            sample_size=(min(silhouette_sample, len(X_scaled))
                         if silhouette_sample else None),
            random_state=config.RANDOM_STATE,
        )
        shares = np.bincount(km.labels_, minlength=k) / len(km.labels_)
        rows.append({"k": int(k), "inertia": float(km.inertia_),
                     "silhouette": float(sil),
                     "min_cluster_share": float(shares.min())})
    return pd.DataFrame(rows)


def elbow_k(sweep: pd.DataFrame) -> int:
    """Pick the elbow of the inertia curve with the Kneedle method.

    Both axes are scaled to 0..1. Inertia falls as k grows, so we draw a
    straight line from the first point to the last and choose the k whose
    inertia sits furthest *below* that line - the point where adding more
    clusters stops paying off.
    """
    k = sweep["k"].to_numpy(dtype=float)
    inertia = sweep["inertia"].to_numpy(dtype=float)
    if len(k) < 3:
        return int(k[0])
    x = (k - k.min()) / (k.max() - k.min())
    span = inertia.max() - inertia.min()
    y = (inertia - inertia.min()) / span if span > 0 else np.zeros_like(inertia)
    gap = (1 - x) - y       # distance below the line from (0,1) to (1,0)
    return int(k[int(np.argmax(gap))])


def is_continuous(series: pd.Series) -> bool:
    """True when a feature has enough distinct values to count as continuous."""
    return series.nunique() >= config.CONTINUOUS_MIN_UNIQUE


@dataclass
class Candidate:
    """The result of applying steps 1-3 of the selection rule to one feature set."""
    name: str
    features: list[str]
    sweep: pd.DataFrame
    elbow_k: int
    silhouette_at_elbow: float
    min_share_at_elbow: float
    eligible: bool
    passes_gate: bool
    notes: list[str] = field(default_factory=list)


def evaluate_candidate(name: str, features: list[str], df_features: pd.DataFrame,
                       sweep: pd.DataFrame) -> Candidate:
    """Apply steps 1-3 of the selection rule (see config.py) to one feature set."""
    k = elbow_k(sweep)
    row = sweep.set_index("k").loc[k]
    has_continuous = any(is_continuous(df_features[f]) for f in features)
    sil, share = float(row["silhouette"]), float(row["min_cluster_share"])
    notes = []
    if not has_continuous:
        notes.append("all features are discrete -> not eligible")
    if sil < config.MIN_SILHOUETTE:
        notes.append(f"silhouette {sil:.3f} < {config.MIN_SILHOUETTE}")
    if share < config.MIN_CLUSTER_SHARE:
        notes.append(f"smallest cluster {share:.1%} < {config.MIN_CLUSTER_SHARE:.0%}")
    return Candidate(
        name=name, features=list(features), sweep=sweep, elbow_k=k,
        silhouette_at_elbow=sil, min_share_at_elbow=share,
        eligible=has_continuous,
        passes_gate=(sil >= config.MIN_SILHOUETTE
                     and share >= config.MIN_CLUSTER_SHARE),
        notes=notes,
    )


def choose(candidates: list[Candidate], primary: str) -> tuple[Candidate, int, str]:
    """Apply steps 4-5 of the selection rule.

    Returns (chosen candidate, chosen k, plain-English reason).
    """
    ok = [c for c in candidates if c.eligible and c.passes_gate]
    if ok:
        best_sil = max(c.silhouette_at_elbow for c in ok)
        tied = [c for c in ok
                if best_sil - c.silhouette_at_elbow <= config.SILHOUETTE_TIE]
        best = min(tied, key=lambda c: (len(c.features), c.name))
        reason = (
            f"'{best.name}' has the highest silhouette ({best.silhouette_at_elbow:.3f}) "
            f"at its elbow k={best.elbow_k} among eligible feature sets that pass "
            f"the quality gate (silhouette >= {config.MIN_SILHOUETTE}, smallest "
            f"cluster >= {config.MIN_CLUSTER_SHARE:.0%})."
        )
        return best, best.elbow_k, reason

    fallback = next(c for c in candidates if c.name == primary)
    k = int(fallback.sweep.loc[fallback.sweep["silhouette"].idxmax(), "k"])
    reason = (
        "No eligible feature set passed the quality gate, so the primary set "
        f"'{primary}' is used at its max-silhouette k={k}. Segments are weak."
    )
    return fallback, k, reason


# ---------------------------------------------------------------------------
# 2. Auto-labelling (no hand-picked names)
# ---------------------------------------------------------------------------
# For each feature: the word used when a cluster is clearly above / below the
# overall mean. Unknown features fall back to "High <name>" / "Low <name>".
TRAIT_WORDS = {
    "purchase_rate_per_year": ("High-Frequency", "Low-Frequency"),
    "frequency": ("High-Frequency", "Low-Frequency"),
    "purchase_amount": ("Big-Spender", "Budget"),
    "amount": ("Big-Spender", "Budget"),
    "monetary": ("Big-Spender", "Low-Value"),
    "avg_order_value": ("Big-Basket", "Small-Basket"),
    "discount_flag": ("Discount-Driven", "Full-Price"),
    "discount": ("Discount-Driven", "Full-Price"),
    "discount_share": ("Discount-Driven", "Full-Price"),
    "previous_purchases": ("Long-History", "New"),
    "recency_days": ("Dormant", "Recent"),
    "tenure_days": ("Long-Tenure", "Newcomer"),
    "rating": ("Happy", "Unhappy"),
    "age": ("Older", "Younger"),
}


def _trait_word(feature: str, high: bool) -> str:
    words = TRAIT_WORDS.get(feature)
    if words:
        return words[0] if high else words[1]
    pretty = feature.replace("_", " ").title()
    return f"High {pretty}" if high else f"Low {pretty}"


def cluster_z_scores(df_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Each cluster's mean per feature, in standard deviations from the overall mean."""
    std = df_features.std(ddof=0).replace(0, 1)
    means = df_features.groupby(labels).mean()
    return (means - df_features.mean()) / std


def _retail_k3_names(z: pd.DataFrame, means: pd.DataFrame) -> dict[int, str] | None:
    """The PRD naming for k=3 retail clusters, used only if the data supports it.

    Rule: the cluster with the highest purchase rate is 'High-Frequency
    Shoppers' (only if its rate is clearly above average, z >= TRAIT_Z); of the
    other two, the one with more discount use is 'Discount-Driven Regular' and
    the other 'Full-Price Loyal' (only if their discount use clearly differs).
    Returns None when the profile does not match, so generic names are used.
    """
    needed = {"purchase_rate_per_year", "discount_flag"}
    if len(z) != 3 or not needed.issubset(z.columns):
        return None
    rate_order = means["purchase_rate_per_year"].sort_values(ascending=False)
    top = rate_order.index[0]
    if z.loc[top, "purchase_rate_per_year"] < config.TRAIT_Z:
        return None
    rest = means.drop(index=top)["discount_flag"].sort_values(ascending=False)
    hi_disc, lo_disc = rest.index[0], rest.index[1]
    if z.loc[hi_disc, "discount_flag"] - z.loc[lo_disc, "discount_flag"] < config.TRAIT_Z:
        return None
    return {int(top): "High-Frequency Shoppers",
            int(hi_disc): "Discount-Driven Regular",
            int(lo_disc): "Full-Price Loyal"}


def _rfm_name(zrow: pd.Series) -> str | None:
    """Classic RFM reading, derived from z-scores (low recency = recent)."""
    cols = {"recency_days", "frequency", "monetary"}
    if not cols.issubset(zrow.index):
        return None
    r, f, m = zrow["recency_days"], zrow["frequency"], zrow["monetary"]
    t = config.TRAIT_Z
    if r <= -t and f >= t and m >= t:
        return "Champions"
    if r <= -t and (f >= t or m >= t):
        return "Loyal Active"
    if r >= t and (f >= t or m >= t):
        return "At-Risk (was valuable)"
    if r >= t and f <= 0:
        return "At-Risk / Dormant"
    if r <= -t and f < t:
        return "New / Promising"
    return None


def label_clusters(df_features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Give every cluster a descriptive, unique name from its own profile.

    Order of preference:
      1. The PRD retail names for a k=3 frequency/discount split (if it fits).
      2. RFM names (Champions, At-Risk / Dormant, ...) for RFM features.
      3. Generic trait names: the one or two features where the cluster is
         furthest from the overall mean (|z| >= TRAIT_Z), e.g.
         'High-Frequency · Discount-Driven'. Clusters with no strong trait are
         'Mid-Range'.
    Duplicate names get a short distinguishing suffix. Never 'Cluster 0'.
    The mapping depends only on cluster profiles, so it is stable across runs.
    """
    z = cluster_z_scores(df_features, labels)
    means = df_features.groupby(labels).mean()

    retail = _retail_k3_names(z, means)
    if retail:
        return retail

    names: dict[int, str] = {}
    for cluster, zrow in z.iterrows():
        name = _rfm_name(zrow)
        if name is None:
            strong = zrow[zrow.abs() >= config.TRAIT_Z]
            strong = strong.reindex(strong.abs().sort_values(ascending=False).index)
            traits = [_trait_word(f, v > 0) for f, v in strong.head(2).items()]
            name = " · ".join(traits) if traits else "Mid-Range"
        names[int(cluster)] = name

    # Make names unique: add the next most distinctive trait, then a number.
    counts = pd.Series(list(names.values())).value_counts()
    for dup in counts[counts > 1].index:
        members = [c for c, n in names.items() if n == dup]
        for i, c in enumerate(sorted(members, key=lambda c: -z.loc[c].abs().sum())):
            zrow = z.loc[c].reindex(z.loc[c].abs().sort_values(ascending=False).index)
            extra = _trait_word(zrow.index[0], zrow.iloc[0] > 0)
            candidate = f"{dup} ({extra})" if extra not in dup else f"{dup} ({i + 1})"
            names[c] = candidate
    counts = pd.Series(list(names.values())).value_counts()
    for dup in counts[counts > 1].index:
        for i, c in enumerate(sorted(c for c, n in names.items() if n == dup)):
            names[c] = f"{dup} #{i + 1}"
    return names


# ---------------------------------------------------------------------------
# 3. Segment profile + plain-English description
# ---------------------------------------------------------------------------
def segment_profile(df: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """Summary table, one row per segment, using whichever columns exist.

    Columns (when available): customers, pct_of_base, avg_purchase_rate_per_year,
    avg_spend, discount_usage_pct, avg_previous_purchases, subscriber_pct,
    top_category, plus RFM averages for transaction data.
    """
    g = df.groupby(segment_col, observed=True)
    prof = pd.DataFrame({"customers": g.size()})
    prof["pct_of_base"] = 100 * prof["customers"] / prof["customers"].sum()
    numeric = {
        "purchase_rate_per_year": "avg_purchase_rate_per_year",
        "purchase_amount": "avg_spend",
        "previous_purchases": "avg_previous_purchases",
        "recency_days": "avg_recency_days",
        "frequency": "avg_orders",
        "monetary": "avg_monetary",
        "avg_order_value": "avg_order_value",
    }
    for col, out in numeric.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            prof[out] = g[col].mean()
    if "discount_flag" in df.columns:
        prof["discount_usage_pct"] = 100 * g["discount_flag"].mean()
    elif "discount_share" in df.columns:
        prof["discount_usage_pct"] = 100 * g["discount_share"].mean()
    if "subscription_status" in df.columns:
        prof["subscriber_pct"] = 100 * g["subscription_status"].apply(
            lambda s: (s == "Yes").mean())
    if "category" in df.columns:
        prof["top_category"] = g["category"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    return prof.sort_values("customers", ascending=False).round(2)


def describe_segment(row: pd.Series) -> str:
    """One plain-English line built from a segment_profile() row."""
    parts = []
    if "avg_purchase_rate_per_year" in row:
        parts.append(f"buys ~{row['avg_purchase_rate_per_year']:.0f}×/yr")
    if "avg_orders" in row:
        parts.append(f"{row['avg_orders']:.1f} orders on average")
    if "avg_recency_days" in row:
        parts.append(f"last bought ~{row['avg_recency_days']:.0f} days ago")
    if "discount_usage_pct" in row:
        parts.append(f"{row['discount_usage_pct']:.0f}% use a discount")
    if "avg_spend" in row:
        parts.append(f"avg spend ${row['avg_spend']:.2f}")
    if "avg_monetary" in row:
        parts.append(f"total spend ${row['avg_monetary']:.2f}")
    if "avg_previous_purchases" in row:
        parts.append(f"{row['avg_previous_purchases']:.0f} previous purchases")
    if "top_category" in row and pd.notna(row["top_category"]):
        parts.append(f"mostly {row['top_category']}")
    text = ", ".join(parts)
    return text[:1].upper() + text[1:] if text else ""


# ---------------------------------------------------------------------------
# 4. Prediction with the frozen model
# ---------------------------------------------------------------------------
def predict(df_raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Assign every row of a raw upload to an existing segment.

    Steps: clean -> engineer_features -> build_features -> scaler.transform ->
    kmeans.predict -> labels_map. The model is NEVER refitted: new data is
    placed in the segments learned from the training data.
    """
    df = pipeline.prepare(
        df_raw,
        rating_medians=artifact.get("rating_medians"),
        global_rating_median=artifact.get("global_rating_median"),
        age_bin_edges=artifact.get("age_bin_edges"),
    )
    X = pipeline.build_features(df, artifact["feature_names"])
    ids = artifact["kmeans"].predict(artifact["scaler"].transform(X.to_numpy()))
    df["segment_id"] = ids
    df["segment"] = pd.Series(ids, index=df.index).map(artifact["labels_map"])
    return df


def training_stats(X: pd.DataFrame) -> dict:
    """Mean / std / min / max of each training feature (for drift warnings)."""
    return {c: {"mean": float(X[c].mean()), "std": float(X[c].std(ddof=0)),
                "min": float(X[c].min()), "max": float(X[c].max())}
            for c in X.columns}


def out_of_distribution_warnings(X_new: pd.DataFrame, stats: dict,
                                 z_limit: float = 1.0) -> list[str]:
    """Plain-English warnings when an upload looks unlike the training data.

    A feature is flagged when its mean is more than `z_limit` training standard
    deviations away from the training mean, or when more than 10% of its
    values fall outside the training min..max range.
    """
    warnings = []
    for col, s in stats.items():
        if col not in X_new.columns or len(X_new) == 0:
            continue
        values = X_new[col]
        shift = abs(values.mean() - s["mean"]) / (s["std"] or 1)
        outside = ((values < s["min"]) | (values > s["max"])).mean()
        if shift > z_limit:
            warnings.append(
                f"'{col}': average {values.mean():.2f} vs {s['mean']:.2f} in "
                f"training ({shift:.1f} standard deviations away).")
        if outside > 0.10:
            warnings.append(
                f"'{col}': {outside:.0%} of values are outside the training "
                f"range {s['min']:.0f}–{s['max']:.0f}.")
    return warnings
