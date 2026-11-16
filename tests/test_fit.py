"""Tests for src/fit.py ('Fit on my data')."""
import pandas as pd
import pytest

from src import config, fit, mapping, pipeline, rfm


@pytest.fixture(scope="module")
def artifact():
    import joblib
    if not config.MODEL_PATH.exists():
        pytest.skip("Run `python -m src.train` first")
    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def tx_rfm():
    path = config.DATA_DIR / "sample_transactions.csv"
    if not path.exists():
        pytest.skip("Run `python -m tests.make_synthetic_transactions` first")
    tx = pd.read_csv(path)
    return rfm.build_rfm(mapping.apply_mapping(tx, mapping.suggest_mapping(tx)).df).df


def test_same_k_and_segments_as_train(artifact):
    df = pipeline.engineer_features(pipeline.clean(pipeline.load_raw()))
    res = fit.fit_segmentation(df, features=artifact["feature_names"])
    assert res.k == artifact["kmeans"].n_clusters
    clustered = pd.read_csv(config.CLUSTERED_CSV_PATH)
    assert (res.data["segment"].to_numpy() == clustered["segment"].to_numpy()).all()


def test_rfm_labels_unique_and_descriptive(tx_rfm):
    res = fit.fit_segmentation(tx_rfm)
    names = list(res.labels_map.values())
    assert len(names) == len(set(names)) == res.k
    assert not any(n.lower().startswith("cluster") for n in names)
    assert res.log_features == ["frequency", "monetary"]


def test_refuses_tiny_input(tx_rfm):
    with pytest.raises(ValueError, match="At least"):
        fit.fit_segmentation(tx_rfm.head(20))


def test_needs_two_features():
    df = pd.DataFrame({"purchase_amount": range(100)})
    with pytest.raises(ValueError, match="at least 2"):
        fit.fit_segmentation(df)


def test_does_not_touch_artifacts(tx_rfm):
    before = {p: p.stat().st_mtime for p in config.ARTIFACTS_DIR.rglob("*") if p.is_file()}
    res = fit.fit_segmentation(tx_rfm)
    res.model_bytes(), res.metrics_json()
    after = {p: p.stat().st_mtime for p in config.ARTIFACTS_DIR.rglob("*") if p.is_file()}
    assert before == after
