"""One dashboard implementation for every file.

build_dashboard(df, schema) looks at which roles exist and builds only the KPI
cards and Plotly charts the data supports. Missing roles produce a short note
instead of a chart, so the dashboard never crashes on a missing column.

Page 1 calls it with the fixed retail schema; page 4 calls it with the user's
column mapping. This module has no Streamlit code, so it is easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Theme colours (match the Power BI dashboard).
NAVY = "#1424A0"
PINK = "#E468B9"
PURPLE = "#7B2D9B"
PALETTE = [NAVY, PINK, PURPLE, "#F2A541", "#2BA84A", "#00A6C8", "#8C564B",
           "#555555"]

MAX_FILTER_VALUES = 20
TOP_N_LOCATIONS = 10

# The retail dataset's schema: role -> column in clustered_customers.csv.
RETAIL_SCHEMA = {
    "customer_id": "customer_id",
    "amount": "purchase_amount",
    "category": "category",
    "item": "item_purchased",
    "gender": "gender",
    "age": "age",
    "rating": "review_rating",
    "subscription": "subscription_status",
    "shipping": "shipping_type",
    "location": "location",
    "discount": "discount_applied",
    "previous_purchases": "previous_purchases",
    "frequency": "frequency_of_purchases",
}


@dataclass
class Chart:
    key: str
    section: str
    title: str
    fig: go.Figure


@dataclass
class Dashboard:
    kpis: list[tuple[str, str]] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def chart_keys(self) -> list[str]:
        return [c.key for c in self.charts]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def filter_columns(df: pd.DataFrame, schema: dict) -> list[str]:
    """Low-cardinality categorical columns that get a sidebar filter."""
    candidates = [schema.get(r) for r in ("subscription", "gender", "category",
                                           "shipping")]
    candidates += ["age_group", "segment"]
    out = []
    for col in candidates:
        if col and col in df.columns and 1 < df[col].nunique() <= MAX_FILTER_VALUES:
            out.append(col)
    return out


def apply_filters(df: pd.DataFrame, selections: dict[str, list],
                  date_col: str | None = None, date_range=None) -> pd.DataFrame:
    """Keep rows whose values are in every selected list (empty list = all)."""
    mask = pd.Series(True, index=df.index)
    for col, values in selections.items():
        if values and col in df.columns:
            mask &= df[col].astype(str).isin([str(v) for v in values])
    if date_col and date_range and date_col in df.columns:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        d = df[date_col]
        mask &= (d >= start) & (d < end + pd.Timedelta(days=1))
    return df[mask]


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color="#222"), legend_title_text="")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _bar(data: pd.DataFrame, x: str, y: str, title: str, horizontal=False,
         color=NAVY, fmt=",.0f") -> go.Figure:
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation="h", text=y, title=title)
        fig.update_yaxes(categoryorder="total ascending", title="")
    else:
        fig = px.bar(data, x=x, y=y, text=y, title=title)
        fig.update_xaxes(title="")
    fig.update_traces(marker_color=color, texttemplate=f"%{{text:{fmt}}}",
                      textposition="outside", cliponaxis=False)
    return _style(fig)


def _donut(counts: pd.Series, title: str) -> go.Figure:
    fig = px.pie(names=counts.index.astype(str), values=counts.values, hole=0.55,
                 title=title, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="percent+label")
    return _style(fig)


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_dashboard(df: pd.DataFrame, schema: dict,
                    segment_col: str = "segment") -> Dashboard:
    """Build KPIs + charts for whatever roles `schema` maps in `df`.

    Args:
        df: canonical data (one row per purchase or per customer).
        schema: {role: column name in df}; roles that are absent are skipped.
        segment_col: column holding segment names, if any.
    """
    d = Dashboard()
    col = {r: c for r, c in schema.items() if c and c in df.columns}
    amount = col.get("amount")
    has_seg = segment_col in df.columns and df[segment_col].notna().any()

    if df.empty:
        d.notes.append("No rows match the current filters.")
        return d

    # ---------------- KPIs ----------------
    n_customers = df[col["customer_id"]].nunique() if "customer_id" in col else len(df)
    d.kpis.append(("Customers", f"{n_customers:,}"))
    if "transaction_date" in col:
        d.kpis.append(("Transactions", f"{len(df):,}"))
    if amount:
        d.kpis.append(("Total Revenue", _fmt_money(df[amount].sum())))
        d.kpis.append(("Avg Purchase Amount", _fmt_money(df[amount].mean())))
    if "rating" in col:
        d.kpis.append(("Avg Review Rating", f"{df[col['rating']].mean():.2f}"))
    disc = "discount_flag" if "discount_flag" in df.columns else None
    if disc:
        d.kpis.append(("Discount Usage", f"{100 * df[disc].mean():.1f}%"))
    if "subscription" in col:
        d.kpis.append(("Subscribers",
                       f"{100 * (df[col['subscription']] == 'Yes').mean():.1f}%"))

    # ---------------- Customers ----------------
    if "subscription" in col:
        counts = df[col["subscription"]].value_counts()
        d.charts.append(Chart("subscription_donut", "Customers",
                              "% customers by subscription status",
                              _donut(counts, "Customers by subscription status")))
    else:
        d.notes.append("Add a Subscription column to see the subscription chart.")

    if "gender" in col and amount:
        g = df.groupby(col["gender"])[amount].sum().reset_index()
        d.charts.append(Chart("revenue_by_gender", "Customers", "Revenue by gender",
                              _bar(g, col["gender"], amount, "Revenue by gender",
                                   color=PURPLE)))
    elif amount:
        d.notes.append("Add a Gender column to see revenue by gender.")

    if "age_group" in df.columns and df["age_group"].notna().any() and amount:
        order = sorted(df["age_group"].dropna().unique())
        ag = df.groupby("age_group")[amount].agg(["sum", "count"]).reset_index()
        ag.columns = ["age_group", "revenue", "sales"]
        for metric, title in [("revenue", "Revenue by age group"),
                              ("sales", "Sales (count) by age group")]:
            fig = _bar(ag, "age_group", metric, title, horizontal=True)
            fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
            d.charts.append(Chart(f"{metric}_by_age_group", "Customers", title, fig))
    else:
        d.notes.append("Add an Age column to see age-group charts.")

    if "location" in col and amount:
        loc = (df.groupby(col["location"])[amount].sum().nlargest(TOP_N_LOCATIONS)
               .reset_index())
        d.charts.append(Chart("revenue_by_location", "Customers",
                              f"Top {TOP_N_LOCATIONS} locations by revenue",
                              _bar(loc, col["location"], amount,
                                   f"Top {TOP_N_LOCATIONS} locations by revenue",
                                   horizontal=True)))

    # ---------------- Products ----------------
    if "category" in col and amount:
        cat = df.groupby(col["category"])[amount].agg(["sum", "count"]).reset_index()
        cat.columns = ["category", "revenue", "sales"]
        d.charts.append(Chart("revenue_by_category", "Products", "Revenue by category",
                              _bar(cat, "category", "revenue", "Revenue by category")))
        d.charts.append(Chart("sales_by_category", "Products",
                              "Sales (count) by category",
                              _bar(cat, "category", "sales", "Sales (count) by category",
                                   color=PINK)))
    else:
        d.notes.append("Add a Category column to see category charts.")

    if "item" in col and "rating" in col:
        top = (df.groupby(col["item"])[col["rating"]].mean().nlargest(5)
               .reset_index())
        d.charts.append(Chart("top_rated_products", "Products",
                              "Top 5 products by average rating",
                              _bar(top, col["item"], col["rating"],
                                   "Top 5 products by average rating",
                                   horizontal=True, color=PURPLE, fmt=".2f")))
    if "item" in col and disc:
        dd = (df.groupby(col["item"])[disc].mean().mul(100).nlargest(5)
              .reset_index())
        d.charts.append(Chart("discount_dependent_products", "Products",
                              "Top 5 discount-dependent products (% bought with discount)",
                              _bar(dd, col["item"], disc,
                                   "Top 5 discount-dependent products (%)",
                                   horizontal=True, color=PINK, fmt=".1f")))
    if "item" not in col:
        d.notes.append("Add an Item/Product column to see product rankings.")

    if "shipping" in col and amount:
        sh = df.groupby(col["shipping"])[amount].mean().reset_index()
        d.charts.append(Chart("avg_spend_by_shipping", "Products",
                              "Average spend by shipping type",
                              _bar(sh, col["shipping"], amount,
                                   "Average spend by shipping type", fmt=",.2f")))

    # ---------------- Trends (dates) ----------------
    if "transaction_date" in col and amount:
        dates = df[col["transaction_date"]]
        monthly = (df.assign(month=dates.dt.to_period("M").dt.to_timestamp())
                   .groupby("month")[amount].agg(["sum", "count"]).reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["sum"],
                                 name="Revenue", line=dict(color=NAVY, width=3)))
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["count"],
                                 name="Orders", yaxis="y2",
                                 line=dict(color=PINK, width=2, dash="dot")))
        fig.update_layout(title="Monthly revenue and orders",
                          yaxis=dict(title="Revenue"),
                          yaxis2=dict(title="Orders", overlaying="y", side="right"))
        d.charts.append(Chart("monthly_trend", "Trends", "Monthly revenue and orders",
                              _style(fig)))
        dow = (df.assign(day=dates.dt.day_name()).groupby("day")[amount].count()
               .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                         "Saturday", "Sunday"]).fillna(0).reset_index())
        dow.columns = ["day", "orders"]
        fig = _bar(dow, "day", "orders", "Orders by day of week", color=PURPLE)
        fig.update_xaxes(categoryorder="array", categoryarray=list(dow["day"]))
        d.charts.append(Chart("orders_by_weekday", "Trends", "Orders by day of week",
                              fig))
    else:
        d.notes.append("Add a Date column to see trends over time.")

    # ---------------- Segments ----------------
    if has_seg:
        counts = df[segment_col].value_counts()
        d.charts.append(Chart("segment_donut", "Segments", "Customers by segment",
                              _donut(counts, "Customers by K-Means segment")))
        if amount:
            rs = df.groupby(segment_col)[amount].sum().reset_index()
            d.charts.append(Chart("revenue_by_segment", "Segments", "Revenue by segment",
                                  _bar(rs, segment_col, amount, "Revenue by segment",
                                       color=PURPLE)))
        if "category" in col:
            ct = pd.crosstab(df[segment_col], df[col["category"]])
            fig = px.imshow(ct, text_auto=True, aspect="auto",
                            color_continuous_scale=["#ffffff", PINK, PURPLE],
                            title="Segment × category (number of purchases)")
            fig.update_xaxes(title="")
            fig.update_yaxes(title="")
            d.charts.append(Chart("segment_x_category", "Segments",
                                  "Segment × category", _style(fig, 380)))
    else:
        d.notes.append("Run a segmentation to see segment charts.")
    return d
