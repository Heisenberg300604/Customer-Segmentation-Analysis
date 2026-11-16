"""Train the retail K-Means model and write every artifact.

Run from the project root:
    python -m src.train

What it does:
  1. Loads and cleans data/Customer_dataset.csv with the shared pipeline.
  2. Runs the k = 2..8 sweep (inertia + silhouette) for the PRD's 4-feature
     design AND the alternative feature sets.
  3. Applies the written selection rule from config.py to choose the feature
     set and k (no tuning towards any target number).
  4. Fits the final model, auto-labels the clusters, builds the segment profile
     and a surrogate decision-tree explanation.
  5. Re-runs the selection with Bi-Weekly = 104/yr to check sensitivity.
  6. Saves clustered_customers.csv, kmeans_model.joblib, metrics.json and the
     elbow / silhouette / PCA figures.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from src import config, pipeline, segmentation  # noqa: E402

MODEL_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_training_data() -> tuple[pd.DataFrame, dict]:
    """Load + clean + engineer the training data; return it with the stored
    preprocessing parameters (rating medians and age bin edges)."""
    raw = pipeline.load_raw(config.RAW_DATA_PATH)

    # Medians are taken from the ratings that were actually given (before
    # imputation), exactly as clean() does internally for training data.
    ratings = pipeline.normalise_column_names(raw)
    ratings["review_rating"] = pd.to_numeric(ratings["review_rating"],
                                             errors="coerce")
    medians, global_median = pipeline.rating_medians_by_category(ratings)

    df = pipeline.engineer_features(pipeline.clean(raw))
    params = {
        "rating_medians": medians,
        "global_rating_median": global_median,
        "age_bin_edges": pipeline.compute_age_bin_edges(df["age"]),
    }
    return df, params


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def run_selection(df: pd.DataFrame) -> dict:
    """Sweep every feature set, apply the selection rule, fit the final model.

    Returns a dict with the candidates, the chosen set/k/reason, the fitted
    scaler and model, the labels and the label names.
    """
    candidates = []
    k3_sizes = {}
    for name, features in config.FEATURE_SETS.items():
        X = StandardScaler().fit_transform(pipeline.build_features(df, features))
        sweep = segmentation.kmeans_sweep(X, config.K_RANGE)
        candidates.append(segmentation.evaluate_candidate(
            name, features, df[features], sweep))
        km3 = KMeans(3, n_init=config.KMEANS_N_INIT,
                     random_state=config.RANDOM_STATE).fit(X)
        names = segmentation.label_clusters(df[features], km3.labels_)
        sizes = pd.Series(km3.labels_).map(names).value_counts()
        k3_sizes[name] = {str(k): int(v) for k, v in sizes.items()}

    chosen, k, reason = segmentation.choose(candidates, config.PRIMARY_FEATURE_SET)

    scaler = StandardScaler().fit(
        pipeline.build_features(df, chosen.features).to_numpy())
    X = scaler.transform(pipeline.build_features(df, chosen.features).to_numpy())
    kmeans = KMeans(k, n_init=config.KMEANS_N_INIT,
                    random_state=config.RANDOM_STATE).fit(X)
    labels_map = segmentation.label_clusters(df[chosen.features], kmeans.labels_)
    return {"candidates": candidates, "k3_sizes": k3_sizes, "chosen": chosen,
            "k": k, "reason": reason, "scaler": scaler, "kmeans": kmeans,
            "X": X, "labels": kmeans.labels_, "labels_map": labels_map}


def biweekly_sensitivity(df: pd.DataFrame, base: dict) -> dict:
    """Re-run the whole selection with Bi-Weekly = 104/yr and compare."""
    alt = df.copy()
    alt_map = {**config.FREQUENCY_TO_PER_YEAR, "Bi-Weekly": 104}
    alt["purchase_rate_per_year"] = (
        alt["frequency_of_purchases"].map(alt_map).astype(float))
    res = run_selection(alt)
    seg_base = pd.Series(base["labels"]).map(base["labels_map"])
    seg_alt = pd.Series(res["labels"]).map(res["labels_map"])
    return {
        "bi_weekly_per_year_tested": 104,
        "chosen_feature_set": res["chosen"].name,
        "chosen_k": res["k"],
        "silhouette": float(silhouette_score(res["X"], res["labels"])),
        "segment_sizes": {str(k): int(v) for k, v in seg_alt.value_counts().items()},
        "adjusted_rand_index_vs_default": float(
            adjusted_rand_score(seg_base, seg_alt)),
        "same_segments": bool(adjusted_rand_score(seg_base, seg_alt) == 1.0),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)


def plot_sweeps(candidates, chosen_name: str, chosen_k: int) -> None:
    """elbow.png and silhouette.png: one line per feature set, chosen in bold."""
    for metric, fname, ylabel in [
        ("inertia", "elbow.png", "Inertia (within-cluster sum of squares)"),
        ("silhouette", "silhouette.png", "Silhouette score"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in candidates:
            bold = c.name == chosen_name
            ax.plot(c.sweep["k"], c.sweep[metric], marker="o",
                    linewidth=3 if bold else 1.2, alpha=1 if bold else 0.55,
                    label=f"{c.name}{'  (chosen)' if bold else ''}")
            if metric == "inertia":
                e = c.sweep.set_index("k").loc[c.elbow_k, "inertia"]
                ax.scatter([c.elbow_k], [e], s=160, facecolors="none",
                           edgecolors="black", linewidths=1.5, zorder=5)
        if metric == "silhouette":
            ax.axhline(config.MIN_SILHOUETTE, color="grey", linestyle="--",
                       label=f"quality gate ({config.MIN_SILHOUETTE})")
        ax.axvline(chosen_k, color="#7B2D9B", linestyle=":", label=f"chosen k = {chosen_k}")
        title = ("Elbow method (circles = Kneedle elbow per feature set)"
                 if metric == "inertia" else "Silhouette score by k")
        _style_axes(ax, title, "Number of clusters k", ylabel)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / fname, dpi=150)
        plt.close(fig)


def plot_pca(X: np.ndarray, segments: pd.Series) -> dict:
    """pca_clusters.png: 2-D PCA of the scaled features, coloured by segment."""
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    P = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1424A0", "#E468B9", "#7B2D9B", "#F2A541", "#2BA84A",
              "#00A6C8", "#8C564B", "#555555"]
    for i, seg in enumerate(sorted(segments.unique())):
        mask = (segments == seg).to_numpy()
        ax.scatter(P[mask, 0], P[mask, 1], s=10, alpha=0.5,
                   color=colors[i % len(colors)], label=f"{seg} (n={mask.sum():,})")
    _style_axes(ax, "Customer segments – PCA projection of scaled features",
                f"PC1 ({ev[0]:.1%} of variance)", f"PC2 ({ev[1]:.1%} of variance)")
    ax.legend(fontsize=10, markerscale=2)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pca_clusters.png", dpi=150)
    plt.close(fig)
    return {"explained_variance_ratio": [float(v) for v in ev]}


# ---------------------------------------------------------------------------
# Surrogate explanation
# ---------------------------------------------------------------------------
def surrogate_tree(df_features: pd.DataFrame, segments: pd.Series) -> dict:
    """Train a depth-3 decision tree to mimic the clusters (a SURROGATE).

    K-Means has no built-in feature importance, so this tree approximates
    which features separate the segments. `fidelity` = how often the tree
    agrees with K-Means; it is an explanation aid, not the model itself.
    """
    tree = DecisionTreeClassifier(max_depth=config.SURROGATE_MAX_DEPTH,
                                  random_state=config.RANDOM_STATE)
    tree.fit(df_features, segments)
    return {
        "note": "Surrogate explanation: a decision tree trained to imitate the "
                "K-Means segments. Not the segmentation model itself.",
        "fidelity": float(tree.score(df_features, segments)),
        "feature_importances": {
            f: float(v) for f, v in zip(df_features.columns, tree.feature_importances_)},
        "rules": export_text(tree, feature_names=list(df_features.columns),
                             decimals=1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, params = load_training_data()
    res = run_selection(df)
    chosen, k = res["chosen"], res["k"]

    # Assign segments and save the clustered dataset.
    df["segment_id"] = res["labels"]
    df["segment"] = df["segment_id"].map(res["labels_map"])
    df.to_csv(config.CLUSTERED_CSV_PATH, index=False)

    final_sil = float(silhouette_score(res["X"], res["labels"]))
    profile = segmentation.segment_profile(df)
    profile["description"] = profile.apply(segmentation.describe_segment, axis=1)

    plot_sweeps(res["candidates"], chosen.name, k)
    pca_info = plot_pca(res["X"], df["segment"])
    surrogate = surrogate_tree(df[chosen.features], df["segment"])
    sensitivity = biweekly_sensitivity(df, res)

    X_chosen = pipeline.build_features(df, chosen.features)
    artifact = {
        "scaler": res["scaler"],
        "kmeans": res["kmeans"],
        "labels_map": res["labels_map"],
        "feature_names": chosen.features,
        "feature_set_name": chosen.name,
        "age_bin_edges": params["age_bin_edges"],
        "rating_medians": params["rating_medians"],
        "global_rating_median": params["global_rating_median"],
        "training_stats": segmentation.training_stats(X_chosen),
        "config_snapshot": {
            "frequency_to_per_year": config.FREQUENCY_TO_PER_YEAR,
            "random_state": config.RANDOM_STATE,
            "k_range": list(config.K_RANGE),
            "n_init": config.KMEANS_N_INIT,
            "min_silhouette": config.MIN_SILHOUETTE,
            "min_cluster_share": config.MIN_CLUSTER_SHARE,
        },
        "sklearn_version": sklearn.__version__,
        "version": MODEL_VERSION,
    }
    joblib.dump(artifact, config.MODEL_PATH)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "selection_rule": "See config.py: Kneedle elbow per feature set -> "
                          "quality gate -> continuous-feature eligibility -> "
                          "highest silhouette.",
        "feature_sets": {
            c.name: {
                "features": c.features,
                "sweep": c.sweep.round(4).to_dict(orient="records"),
                "elbow_k": c.elbow_k,
                "silhouette_at_elbow": round(c.silhouette_at_elbow, 4),
                "min_cluster_share_at_elbow": round(c.min_share_at_elbow, 4),
                "eligible": c.eligible,
                "passes_gate": c.passes_gate,
                "notes": c.notes,
                "segment_sizes_at_k3": res["k3_sizes"][c.name],
            } for c in res["candidates"]},
        "chosen": {
            "feature_set": chosen.name,
            "features": chosen.features,
            "k": k,
            "silhouette": round(final_sil, 4),
            "reason": res["reason"],
        },
        "segment_profile": profile.reset_index().to_dict(orient="records"),
        "pca": pca_info,
        "surrogate_tree": surrogate,
        "bi_weekly_sensitivity": sensitivity,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    _print_summary(metrics, profile)
    return metrics


def _print_summary(metrics: dict, profile: pd.DataFrame) -> None:
    """Human-readable report of the real numbers."""
    print("\n=== k-sweep: silhouette by k ===")
    table = pd.DataFrame({
        name: {r["k"]: r["silhouette"] for r in fs["sweep"]}
        for name, fs in metrics["feature_sets"].items()})
    print(table.round(3).to_string())
    print("\n=== Selection rule, per feature set ===")
    for name, fs in metrics["feature_sets"].items():
        print(f"{name:15s} elbow k={fs['elbow_k']}  sil={fs['silhouette_at_elbow']:.3f}  "
              f"min share={fs['min_cluster_share_at_elbow']:.1%}  eligible={fs['eligible']}  "
              f"gate={fs['passes_gate']}  {'; '.join(fs['notes'])}")
    print("\n=== Segment sizes at k=3 per feature set ===")
    for name, fs in metrics["feature_sets"].items():
        print(f"{name:15s} {fs['segment_sizes_at_k3']}")
    ch = metrics["chosen"]
    print(f"\n=== CHOSEN: {ch['feature_set']} {ch['features']}, k={ch['k']}, "
          f"silhouette={ch['silhouette']:.3f}")
    print(ch["reason"])
    print("\n=== Segment profile ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(profile.drop(columns="description").to_string())
    for seg, d in profile["description"].items():
        print(f"  - {seg}: {d}")
    s = metrics["surrogate_tree"]
    print(f"\nSurrogate tree fidelity: {s['fidelity']:.3f}; importances: "
          f"{ {k: round(v, 3) for k, v in s['feature_importances'].items()} }")
    print(f"\nBi-Weekly=104 sensitivity: {metrics['bi_weekly_sensitivity']}")


if __name__ == "__main__":
    main()
