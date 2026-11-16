"""Tests for src/segmentation.py and the trained artifacts.

Artifact tests need `python -m src.train` to have been run (they skip otherwise).
"""
import joblib
import numpy as np
import pandas as pd
import pytest

from src import config, pipeline, segmentation


@pytest.fixture(scope="module")
def artifact():
    if not config.MODEL_PATH.exists():
        pytest.skip("Run `python -m src.train` first")
    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def clustered():
    if not config.CLUSTERED_CSV_PATH.exists():
        pytest.skip("Run `python -m src.train` first")
    return pd.read_csv(config.CLUSTERED_CSV_PATH)


@pytest.fixture(scope="module")
def raw():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("dataset missing")
    return pipeline.load_raw()


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------
def test_elbow_k_finds_the_bend():
    # Inertia drops fast until k=4, then flattens -> elbow at 4.
    sweep = pd.DataFrame({"k": [2, 3, 4, 5, 6, 7, 8],
                          "inertia": [100, 60, 30, 27, 25, 24, 23]})
    assert segmentation.elbow_k(sweep) == 4


def test_label_clusters_unique_and_never_generic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"purchase_amount": rng.normal(50, 10, 300),
                      "discount_flag": rng.integers(0, 2, 300)})
    labels = rng.integers(0, 5, 300)  # random clusters: names may collide
    names = segmentation.label_clusters(X, labels)
    assert len(set(names.values())) == len(names) == 5
    assert not any(n.lower().startswith("cluster") for n in names.values())


def test_retail_k3_names_when_profile_matches():
    X = pd.DataFrame({
        "purchase_rate_per_year": [52] * 10 + [12] * 20,
        "discount_flag": [1, 0] * 5 + [1] * 10 + [0] * 10,
    })
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    assert segmentation.label_clusters(X, labels) == {
        0: "High-Frequency Shoppers", 1: "Discount-Driven Regular",
        2: "Full-Price Loyal"}


def test_rfm_names():
    X = pd.DataFrame({"recency_days": [5] * 10 + [300] * 10,
                      "frequency": [20] * 10 + [1] * 10,
                      "monetary": [900] * 10 + [30] * 10})
    names = segmentation.label_clusters(X, np.array([0] * 10 + [1] * 10))
    assert names == {0: "Champions", 1: "Dormant / Lost"}


def test_choose_rejects_all_discrete_sets():
    df = pd.DataFrame({"a": np.arange(100.0), "flag": [0, 1] * 50})
    good = pd.DataFrame({"k": [2, 3, 4], "inertia": [10, 5, 4],
                         "silhouette": [0.4, 0.5, 0.45],
                         "min_cluster_share": [0.3, 0.2, 0.1]})
    great = good.assign(silhouette=[0.9, 0.95, 0.97])
    cands = [segmentation.evaluate_candidate("cont", ["a", "flag"], df, good),
             segmentation.evaluate_candidate("disc", ["flag"], df, great)]
    chosen, k, _ = segmentation.choose(cands, primary="cont")
    assert chosen.name == "cont" and k == 3


def test_ood_warning_triggers():
    stats = {"purchase_amount": {"mean": 60, "std": 20, "min": 20, "max": 100}}
    shifted = pd.DataFrame({"purchase_amount": [500.0] * 20})
    normal = pd.DataFrame({"purchase_amount": [55.0, 65.0] * 10})
    assert segmentation.out_of_distribution_warnings(shifted, stats)
    assert not segmentation.out_of_distribution_warnings(normal, stats)


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------
def test_predict_reproduces_training_segments(artifact, clustered, raw):
    pred = segmentation.predict(raw, artifact)
    assert len(pred) == len(clustered)
    assert (pred["segment"].to_numpy() == clustered["segment"].to_numpy()).all()


def test_predict_on_50_row_sample(artifact, raw):
    sample = raw.sample(50, random_state=1)
    pred = segmentation.predict(sample, artifact)
    assert len(pred) == 50
    assert pred["segment"].notna().all()
    assert set(pred["segment"]).issubset(set(artifact["labels_map"].values()))


def test_predict_never_refits(artifact, raw):
    before = artifact["kmeans"].cluster_centers_.copy()
    segmentation.predict(raw.head(100), artifact)
    assert np.array_equal(before, artifact["kmeans"].cluster_centers_)


def test_labels_unique_per_cluster(artifact):
    names = list(artifact["labels_map"].values())
    assert len(names) == len(set(names)) == artifact["kmeans"].n_clusters


def test_retrain_gives_identical_labels(artifact, clustered):
    from src import train
    df, _ = train.load_training_data()
    res = train.run_selection(df)
    assert res["labels_map"] == artifact["labels_map"]
    segs = pd.Series(res["labels"]).map(res["labels_map"])
    assert (segs.to_numpy() == clustered["segment"].to_numpy()).all()
