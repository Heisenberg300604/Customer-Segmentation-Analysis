"""Page 3 - Segment Explorer: filter the segmented customer base and compare
the filtered group with everyone."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from src import dashboard_engine as de  # noqa: E402

ui.setup("Segment Explorer", "Filter, inspect and export segmented customers")
ui.require_artifacts()
df = ui.load_clustered()

st.sidebar.header("Filters")
selections = {}
for col in ["segment", "gender", "category", "age_group", "subscription_status"]:
    options = sorted(df[col].dropna().astype(str).unique())
    selections[col] = st.sidebar.multiselect(col.replace("_", " ").title(), options,
                                             default=options, key=f"p3_{col}")
search = st.sidebar.text_input("Search Customer ID", key="p3_search",
                               help="Matches any ID containing this text.")

view = de.apply_filters(df, selections)
if search.strip():
    view = view[view["customer_id"].astype(str).str.contains(search.strip(),
                                                              regex=False)]

main, side = st.columns([3, 1])
with main:
    st.markdown(f"**{len(view):,}** of {len(df):,} customers match.")
    cols = ["customer_id", "segment", "gender", "age", "age_group", "category",
            "item_purchased", "purchase_amount", "frequency_of_purchases",
            "discount_applied", "previous_purchases", "subscription_status",
            "review_rating", "location"]
    st.dataframe(view[[c for c in cols if c in view.columns]],
                 use_container_width=True, hide_index=True, height=520)
    st.download_button("Download filtered CSV",
                       view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_customers.csv", mime="text/csv",
                       key="p3_download")

with side:
    st.markdown("**Filtered group vs everyone**")
    if view.empty:
        st.info("No customers match.")
    else:
        rows = [("Customers", f"{len(view):,}", f"{len(df):,}"),
                ("Avg spend", f"${view['purchase_amount'].mean():.2f}",
                 f"${df['purchase_amount'].mean():.2f}"),
                ("Discount use", f"{100 * view['discount_flag'].mean():.1f}%",
                 f"{100 * df['discount_flag'].mean():.1f}%"),
                ("Purchases/yr", f"{view['purchase_rate_per_year'].mean():.1f}",
                 f"{df['purchase_rate_per_year'].mean():.1f}"),
                ("Avg rating", f"{view['review_rating'].mean():.2f}",
                 f"{df['review_rating'].mean():.2f}")]
        st.dataframe(pd.DataFrame(rows, columns=["KPI", "Filtered", "All"]),
                     hide_index=True, use_container_width=True)
        mix = pd.concat([
            view["segment"].value_counts(normalize=True).rename("Filtered"),
            df["segment"].value_counts(normalize=True).rename("All"),
        ], axis=1).fillna(0).mul(100).round(1).reset_index(names="segment")
        mix = mix.melt(id_vars="segment", var_name="group", value_name="% of customers")
        fig = px.bar(mix, x="% of customers", y="segment", color="group",
                     barmode="group", orientation="h",
                     color_discrete_map={"Filtered": de.PINK, "All": de.NAVY})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          legend_title_text="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key="p3_mix")

ui.footer()
