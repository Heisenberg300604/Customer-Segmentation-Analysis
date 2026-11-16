"""Page 1 - Overview Dashboard: the web version of the Power BI dashboard,
plus the K-Means segments and the model-validation evidence."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from src import config, dashboard_engine as de, segmentation  # noqa: E402

ui.setup("Customer Behavior Dashboard",
         "Retail customers · KPIs, breakdowns and K-Means segments")
ui.require_artifacts()

df = ui.load_clustered()
metrics = ui.load_metrics()

filtered = ui.sidebar_filters(df, de.RETAIL_SCHEMA, key="p1")
st.caption(f"Showing {len(filtered):,} of {len(df):,} customers.")

dash = de.build_dashboard(filtered, de.RETAIL_SCHEMA)
ui.render_dashboard(dash, key="p1")

# ---------------- Segment profile ----------------
st.subheader("Segment profile")
if filtered.empty:
    st.info("No customers match the filters.")
else:
    profile = segmentation.segment_profile(filtered)
    profile["description"] = profile.apply(segmentation.describe_segment, axis=1)
    ui.profile_table(profile)

# ---------------- Model validation ----------------
chosen = metrics["chosen"]
with st.expander("Model validation (how the segments were chosen)", expanded=False):
    st.markdown(
        f"**Chosen feature set:** `{chosen['feature_set']}` → "
        f"{', '.join(chosen['features'])}  \n"
        f"**Chosen k:** {chosen['k']}  ·  **Silhouette:** {chosen['silhouette']:.3f}  \n"
        f"**Why:** {chosen['reason']}")
    rows = []
    for name, fs in metrics["feature_sets"].items():
        row = {"feature set": name}
        row.update({f"k={r['k']}": round(r["silhouette"], 3) for r in fs["sweep"]})
        row.update({"elbow k": fs["elbow_k"], "eligible": fs["eligible"],
                    "passes gate": fs["passes_gate"]})
        rows.append(row)
    st.markdown("**Silhouette by k for every feature set tested**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    for col, name, caption in [(c1, "elbow.png", "Elbow method"),
                               (c2, "silhouette.png", "Silhouette by k")]:
        path = config.FIGURES_DIR / name
        if path.exists():
            col.image(str(path), caption=caption, use_container_width=True)
    pca_path = config.FIGURES_DIR / "pca_clusters.png"
    if pca_path.exists():
        st.image(str(pca_path), caption="PCA projection coloured by segment",
                 use_container_width=True)
    s = metrics.get("surrogate_tree")
    if s:
        st.markdown(f"**Surrogate explanation** (a depth-3 decision tree that imitates "
                    f"the clusters; agrees with K-Means on {s['fidelity']:.1%} of rows). "
                    "It is an explanation aid, not the model.")
        imp = pd.Series(s["feature_importances"]).sort_values(ascending=False)
        st.dataframe(imp.rename("importance").round(3), use_container_width=True)
        st.code(s["rules"], language="text")

ui.footer()
