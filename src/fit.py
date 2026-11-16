"""'Fit on my data': run the whole segmentation method on ANY mapped file.

This repeats what train.py does for the retail data - scale, k-sweep, pick k
with the SAME written rule, fit K-Means, auto-label, PCA, profile - but on
whatever numeric features the uploaded file has. Nothing is written to
artifacts/: the result is returned in memory and offered as downloads.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config, segmentation
from src.rfm import RFM_FEATURES, RFM_LOG_FEATURES

# Numeric features fit_segmentation() will use when present, in this order.
CANDIDATE_FEATURES = [
    "purchase_rate_per_year", "purchase_amount", "previous_purchases",
    "discount_flag",
]


@dataclass
class FitResult:
    """Everything produced by one fit_segmentation() run."""
    features: list[str]
    log_features: list[str]
    k: int
    reason: str
    sweep: pd.DataFrame
    silhouette: float
    labels_map: dict[int, str]
    scaler: StandardScaler
    kmeans: KMeans
    data: pd.DataFrame              # input rows + segment_id + segment
    pca: pd.DataFrame               # pc1, pc2, segment
    pca_variance: list[float]
    profile: pd.DataFrame
    n_rows_used: int
    sampled: bool
    warnings: list[str] = field(default_factory=list)

    def metrics(self) -> dict:
        return {"features": self.features, "log_features": self.log_features,
                "k": self.k, "reason": self.reason,
                "silhouette": round(self.silhouette, 4),
                "sweep": self.sweep.round(4).to_dict(orient="records"),
                "segment_sizes": self.data["segment"].value_counts().to_dict(),
                "n_rows_used": self.n_rows_used, "sampled": self.sampled,
                "warnings": self.warnings}

    def model_bytes(self) -> bytes:
        """The fitted model as a downloadable joblib file."""
        buf = io.BytesIO()
        joblib.dump({"scaler": self.scaler, "kmeans": self.kmeans,
                     "labels_map": self.labels_map,
                     "feature_names": self.features,
                     "log_features": self.log_features}, buf)
        return buf.getvalue()

    def metrics_json(self) -> str:
        return json.dumps(self.metrics(), indent=2, default=str)


def default_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pick features (and which to log-transform) from what the data has.

    RFM data -> recency, frequency, monetary (log on frequency and monetary).
    Otherwise -> the retail candidate features that exist in the file.
    """
    if all(c in df.columns for c in RFM_FEATURES):
        return list(RFM_FEATURES), list(RFM_LOG_FEATURES)
    return [c for c in CANDIDATE_FEATURES if c in df.columns], []


def fit_segmentation(
    df: pd.DataFrame,
    features: list[str] | None = None,
    log_features: list[str] | None = None,
) -> FitResult:
    """Learn segments from this DataFrame (one row per customer).

    Raises ValueError with a friendly message when the data cannot be
    segmented (too few rows, fewer than 2 usable numeric features).
    """
    warnings: list[str] = []
    if features is None:
        features, auto_log = default_features(df)
        log_features = auto_log if log_features is None else log_features
    log_features = [f for f in (log_features or []) if f in features]

    if len(df) < config.FIT_MIN_ROWS:
        raise ValueError(f"At least {config.FIT_MIN_ROWS} customers are needed "
                         f"to find segments; this file has {len(df)}.")

    data = df.copy()
    sampled = False
    if len(data) > config.FIT_MAX_ROWS:
        data = data.sample(config.FIT_MAX_ROWS, random_state=config.RANDOM_STATE)
        sampled = True
        warnings.append(f"File has {len(df):,} rows; a random sample of "
                        f"{config.FIT_MAX_ROWS:,} was used.")

    # Usable features: present, numeric, not constant. Rows with a missing
    # feature value are left out of the fit.
    usable = []
    for f in features:
        if f in data.columns and pd.api.types.is_numeric_dtype(data[f]):
            if data[f].nunique(dropna=True) > 1:
                usable.append(f)
            else:
                warnings.append(f"'{f}' has the same value everywhere and was dropped.")
    if len(usable) < 2:
        raise ValueError(
            "Segmentation needs at least 2 usable numeric features. Map more "
            "roles (e.g. frequency, previous purchases, discount) or provide "
            "customer id + date + amount so RFM can be computed. "
            f"Usable now: {usable or 'none'}.")
    before = len(data)
    data = data.dropna(subset=usable)
    if len(data) < before:
        warnings.append(f"{before - len(data):,} row(s) with missing feature "
                        "values were left out.")
    if len(data) < config.FIT_MIN_ROWS:
        raise ValueError(f"Only {len(data)} complete rows; at least "
                         f"{config.FIT_MIN_ROWS} are needed.")

    corr = data[usable].corr().abs()
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if corr.loc[a, b] > config.FIT_CORR_WARN:
                warnings.append(f"'{a}' and '{b}' are highly correlated "
                                f"(|r| = {corr.loc[a, b]:.2f}); they carry "
                                "almost the same information.")

    log_features = [f for f in log_features if f in usable]
    Xdf = data[usable].astype(float).copy()
    for f in log_features:
        Xdf[f] = np.log1p(Xdf[f].clip(lower=0))
    scaler = StandardScaler().fit(Xdf.to_numpy())
    X = scaler.transform(Xdf.to_numpy())

    k_max = min(config.FIT_MAX_K, len(data) // config.FIT_ROWS_PER_K)
    k_values = range(2, max(k_max, 2) + 1)
    sweep = segmentation.kmeans_sweep(X, k_values,
                                      silhouette_sample=config.FIT_SILHOUETTE_SAMPLE)

    # Same written rule as train.py, applied to this single feature set.
    cand = segmentation.evaluate_candidate("uploaded", usable, data[usable], sweep)
    chosen, k, reason = segmentation.choose([cand], primary="uploaded")

    kmeans = KMeans(k, n_init=config.KMEANS_N_INIT,
                    random_state=config.RANDOM_STATE).fit(X)
    labels = kmeans.labels_
    labels_map = segmentation.label_clusters(Xdf, labels)
    data["segment_id"] = labels
    data["segment"] = pd.Series(labels, index=data.index).map(labels_map)

    sil = float(silhouette_score(
        X, labels, sample_size=min(config.FIT_SILHOUETTE_SAMPLE, len(X)),
        random_state=config.RANDOM_STATE))
    if sweep["silhouette"].max() < config.MIN_SILHOUETTE:
        warnings.append(f"The best silhouette is {sweep['silhouette'].max():.2f} "
                        f"(< {config.MIN_SILHOUETTE}): segments are weak and "
                        "overlap; treat them as rough groupings.")
    shares = data["segment"].value_counts(normalize=True)
    for seg, share in shares[shares < config.FIT_SMALL_CLUSTER_WARN].items():
        warnings.append(f"Segment '{seg}' holds only {share:.1%} of customers.")

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    P = pca.fit_transform(X)
    pca_df = pd.DataFrame({"pc1": P[:, 0], "pc2": P[:, 1],
                           "segment": data["segment"].to_numpy()})

    profile = segmentation.segment_profile(data)
    profile["description"] = profile.apply(segmentation.describe_segment, axis=1)

    return FitResult(
        features=usable, log_features=log_features, k=k, reason=reason,
        sweep=sweep, silhouette=sil, labels_map=labels_map, scaler=scaler,
        kmeans=kmeans, data=data, pca=pca_df,
        pca_variance=[float(v) for v in pca.explained_variance_ratio_],
        profile=profile, n_rows_used=len(data), sampled=sampled,
        warnings=warnings)
