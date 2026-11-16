"""Headless smoke tests for every Streamlit page (streamlit.testing AppTest).

AppTest cannot attach files to st.file_uploader, so uploads are exercised
through the built-in sample datasets, which use the same code path.
"""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config

PAGES = Path(__file__).resolve().parent.parent / "app" / "pages"
TIMEOUT = 120

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                reason="Run `python -m src.train` first")


def _page(name: str) -> AppTest:
    at = AppTest.from_file(str(PAGES / name), default_timeout=TIMEOUT)
    return at.run()


def _no_errors(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_entry_point_runs():
    at = AppTest.from_file(str(PAGES.parent / "app.py"),
                           default_timeout=TIMEOUT).run()
    assert not at.exception


def test_overview_dashboard():
    at = _page("1_Overview_Dashboard.py")
    _no_errors(at)
    assert any("Customer Behavior Dashboard" in m.value for m in at.markdown)
    # Filtering to one gender still renders.
    at.multiselect(key="p1_f_gender").set_value(["Female"]).run()
    _no_errors(at)


def test_upload_and_predict_sample_matches_training():
    at = _page("2_Upload_and_Predict.py")
    at.checkbox(key="p2_sample").check().run()
    _no_errors(at)
    shown = next(d.value for d in at.dataframe
                 if {"segment", "customers"} <= set(d.value.columns))
    clustered = pd.read_csv(config.CLUSTERED_CSV_PATH)
    expected = clustered["segment"].value_counts()
    got = dict(zip(shown["segment"], shown["customers"]))
    assert got == expected.to_dict()


def test_segment_explorer_and_search():
    at = _page("3_Segment_Explorer.py")
    _no_errors(at)
    at.text_input(key="p3_search").input("123").run()
    _no_errors(at)


def test_analyze_retail_mode_a_and_b():
    at = _page("4_Analyze_Your_Data.py")
    at.radio(key="p4_source").set_value("Sample: retail customers").run()
    _no_errors(at)
    at.button(key="p4_run").click().run()        # Mode A is the default here
    _no_errors(at)
    assert at.session_state["p4_result"]["mode"] == "A"
    at.radio(key="p4_mode").set_value(at.radio(key="p4_mode").options[1]).run()
    at.button(key="p4_run").click().run()
    _no_errors(at)
    assert at.session_state["p4_result"]["mode"] == "B"


def test_analyze_transactions_rfm():
    at = _page("4_Analyze_Your_Data.py")
    at.radio(key="p4_source").set_value(
        "Sample: transactions (synthetic, different column names)").run()
    _no_errors(at)
    at.button(key="p4_run").click().run()
    _no_errors(at)
    res = at.session_state["p4_result"]
    assert res["mode"] == "B"
    assert set(res["fit"].features) == {"recency_days", "frequency", "monetary"}
