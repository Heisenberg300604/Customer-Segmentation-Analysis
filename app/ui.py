"""Shared Streamlit helpers: theme, KPI cards, data loading, upload reading,
sidebar filters and dashboard rendering. Every page imports this module."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, dashboard_engine as de  # noqa: E402

PROJECT_NAME = "Customer Analytics Platform"
NONE_OPTION = "— none —"

CSS = f"""
<style>
.banner {{background:{de.PINK}; color:white; padding:18px 24px; border-radius:10px;
          margin-bottom:18px;}}
.banner h1 {{color:white; margin:0; font-size:2rem;}}
.banner p {{color:white; margin:4px 0 0 0; opacity:0.95;}}
.kpi-card {{background:#FFFFFF; border-left:6px solid {de.PURPLE}; border-radius:8px;
            padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:10px;}}
.kpi-label {{color:#555; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.03em;}}
.kpi-value {{color:{de.NAVY}; font-size:1.6rem; font-weight:700;}}
.footer {{color:#777; font-size:0.8rem; text-align:center; margin-top:40px;
          border-top:1px solid #eee; padding-top:10px;}}
</style>
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def setup(title: str, subtitle: str = "") -> None:
    """Inject the theme CSS and draw the pink title banner."""
    st.markdown(CSS, unsafe_allow_html=True)
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="banner"><h1>{title}</h1>{sub}</div>',
                unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer">{PROJECT_NAME} · K-Means segmentation + '
                'Streamlit · every number on this page is computed from the data</div>',
                unsafe_allow_html=True)


def kpi_row(kpis: list[tuple[str, str]], per_row: int = 6) -> None:
    """Draw KPI cards, `per_row` per line."""
    for start in range(0, len(kpis), per_row):
        chunk = kpis[start:start + per_row]
        for col, (label, value) in zip(st.columns(per_row), chunk):
            col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                         f'<div class="kpi-value">{value}</div></div>',
                         unsafe_allow_html=True)


def render_dashboard(dash: de.Dashboard, key: str) -> None:
    """KPIs, then charts grouped by section in a two-column grid."""
    kpi_row(dash.kpis)
    for section in ["Customers", "Products", "Segments", "Trends"]:
        charts = [c for c in dash.charts if c.section == section]
        if not charts:
            continue
        st.subheader(section)
        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            for col, chart in zip(cols, charts[i:i + 2]):
                col.plotly_chart(chart.fig, use_container_width=True,
                                 key=f"{key}_{chart.key}")
    if dash.notes:
        with st.expander("Charts hidden for this data"):
            for note in dash.notes:
                st.caption(f"• {note}")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
def artifacts_ready() -> bool:
    return config.MODEL_PATH.exists() and config.CLUSTERED_CSV_PATH.exists()


def require_artifacts() -> None:
    if not artifacts_ready():
        st.error("The trained model was not found. Run `python -m src.train` "
                 "from the project folder, then reload this page.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_clustered() -> pd.DataFrame:
    return pd.read_csv(config.CLUSTERED_CSV_PATH)


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict:
    return joblib.load(config.MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _parse_bytes(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            # sep=None lets pandas sniff the delimiter (comma, semicolon, tab...)
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python",
                               encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode the file: {last_error}")


def read_upload(uploaded) -> tuple[pd.DataFrame | None, str | None]:
    """Read an uploaded CSV/Excel file. Returns (df, None) or (None, error)."""
    data = uploaded.getvalue()
    if len(data) == 0:
        return None, "The file is empty."
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return None, (f"The file is larger than {config.MAX_UPLOAD_MB} MB. "
                      "Please upload a smaller extract.")
    try:
        df = _parse_bytes(data, uploaded.name)
    except Exception as exc:  # show a friendly message, never a stack trace
        return None, f"Could not read the file ({type(exc).__name__}): {exc}"
    if df.empty or len(df.columns) < 2:
        return None, ("The file has no rows or only one column. Check that it "
                      "is a CSV with a header row and a comma/semicolon delimiter.")
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df, None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def sidebar_filters(df: pd.DataFrame, schema: dict, key: str) -> pd.DataFrame:
    """Multiselect filters (default = all) for every low-cardinality column,
    plus a date-range picker when the data has dates."""
    st.sidebar.header("Filters")
    selections = {}
    for col in de.filter_columns(df, schema):
        options = sorted(df[col].dropna().astype(str).unique())
        label = col.replace("_", " ").title()
        selections[col] = st.sidebar.multiselect(label, options, default=options,
                                                 key=f"{key}_f_{col}")
    date_col, date_range = schema.get("transaction_date"), None
    if date_col and date_col in df.columns and df[date_col].notna().any():
        lo, hi = df[date_col].min().date(), df[date_col].max().date()
        picked = st.sidebar.date_input("Date range", value=(lo, hi), min_value=lo,
                                       max_value=hi, key=f"{key}_dates")
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            date_range = picked
    return de.apply_filters(df, selections, date_col, date_range)


# ---------------------------------------------------------------------------
# Model-result widgets
# ---------------------------------------------------------------------------
def sweep_charts(sweep: pd.DataFrame, chosen_k: int, key: str) -> None:
    """Plotly elbow + silhouette charts side by side."""
    c1, c2 = st.columns(2)
    for col, metric, title in [(c1, "inertia", "Elbow: inertia by k"),
                               (c2, "silhouette", "Silhouette by k")]:
        fig = go.Figure(go.Scatter(x=sweep["k"], y=sweep[metric],
                                   mode="lines+markers",
                                   line=dict(color=de.NAVY, width=3)))
        fig.add_vline(x=chosen_k, line_dash="dot", line_color=de.PURPLE,
                      annotation_text=f"chosen k={chosen_k}")
        if metric == "silhouette":
            fig.add_hline(y=config.MIN_SILHOUETTE, line_dash="dash",
                          line_color="grey", annotation_text="quality gate")
        fig.update_layout(title=title, xaxis_title="k", yaxis_title=metric,
                          height=330, margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(fig, use_container_width=True, key=f"{key}_{metric}")


def pca_chart(pca: pd.DataFrame, variance: list[float], key: str) -> None:
    plot = pca.sample(min(len(pca), 5000), random_state=config.RANDOM_STATE)
    fig = px.scatter(plot, x="pc1", y="pc2", color="segment", opacity=0.6,
                     color_discrete_sequence=de.PALETTE,
                     title="Segments in 2-D (PCA of the scaled features)",
                     labels={"pc1": f"PC1 ({variance[0]:.1%} of variance)",
                             "pc2": f"PC2 ({variance[1]:.1%} of variance)"})
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True, key=key)


def profile_table(profile: pd.DataFrame) -> None:
    """Segment profile with a plain-English line per segment."""
    st.dataframe(profile.drop(columns="description", errors="ignore"),
                 use_container_width=True)
    if "description" in profile.columns:
        for seg, text in profile["description"].items():
            st.markdown(f"- **{seg}** — {text}")


def segment_distribution(segments: pd.Series, key: str) -> None:
    counts = segments.value_counts().rename_axis("segment").reset_index(name="customers")
    counts["% of rows"] = (100 * counts["customers"] / counts["customers"].sum()).round(1)
    c1, c2 = st.columns([3, 2])
    fig = px.bar(counts, x="segment", y="customers", text="customers",
                 title="Rows per segment")
    fig.update_traces(marker_color=de.NAVY)
    fig.update_layout(height=340, xaxis_title="")
    c1.plotly_chart(fig, use_container_width=True, key=key)
    c2.dataframe(counts, use_container_width=True, hide_index=True)
