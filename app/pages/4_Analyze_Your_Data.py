"""Page 4 - Analyze Your Data: any file -> map columns -> (apply the trained
model | fit on my data) -> full dashboard.

Flow:
  1. Choose a file (upload, or one of two samples).
  2. Map its columns to roles (pre-filled suggestions with confidence).
  3. If it has customer id + date + amount, build RFM (one row per customer).
  4. Segment: Mode A (frozen retail model) or Mode B (learn from this file).
  5. Dashboard on the mapped data, with segments when available.
Uploaded data stays in memory for this session only; nothing is stored.
"""
import hashlib
import json
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from src import config, dashboard_engine as de, fit, mapping, rfm, segmentation  # noqa: E402

ui.setup("Analyze Your Data",
         "Any customer or transactions file · map columns · segment · dashboard")

SOURCES = {
    "Upload a file": None,
    "Sample: retail customers": config.SAMPLE_UPLOAD_PATH,
    "Sample: transactions (synthetic, different column names)":
        config.DATA_DIR / "sample_transactions.csv",
}

# ---------------------------------------------------------------------------
# 1. Data source
# ---------------------------------------------------------------------------
st.header("1 · Choose your data")
source = st.radio("Data source", list(SOURCES), key="p4_source", horizontal=True)
if SOURCES[source] is None:
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"],
                                key="p4_upload")
    if uploaded is None:
        st.caption("Upload a file, or pick a sample above to try the page.")
        ui.footer()
        st.stop()
    df_raw, error = ui.read_upload(uploaded)
    if error:
        st.error(error)
        st.stop()
    src_key = f"upload:{uploaded.name}:{uploaded.size}"
else:
    df_raw = ui.load_csv(str(SOURCES[source]))
    src_key = f"sample:{SOURCES[source].name}"

st.caption(f"{len(df_raw):,} rows × {len(df_raw.columns)} columns")
with st.expander("Preview the raw file"):
    st.dataframe(df_raw.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Column mapping
# ---------------------------------------------------------------------------
st.header("2 · Map your columns")
st.caption("Each role is pre-filled with our best guess. 🟢 high · 🟡 medium · "
           "🟠 low confidence. Choose “— none —” for roles your file does not have.")

loaded_json = st.file_uploader("Optional: load a saved mapping (JSON)", type=["json"],
                               key=f"p4_mapjson_{src_key}")
saved_map, saved_values, saved_dayfirst = {}, {}, False
if loaded_json is not None:
    try:
        saved_map, saved_values, saved_dayfirst = mapping.mapping_from_json(
            loaded_json.getvalue().decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not read that mapping file: {exc}")

suggestions = mapping.suggest_mapping(df_raw, with_confidence=True)
badge = {"high": "🟢", "medium": "🟡", "low": "🟠", "none": "⚪"}
options = [ui.NONE_OPTION] + list(df_raw.columns)
user_map = {}
grid = st.columns(3)
for i, role in enumerate(mapping.ROLES):
    sug = suggestions[role]
    default = saved_map.get(role) if loaded_json is not None else sug.column
    index = options.index(default) if default in options else 0
    required = " *" if role in mapping.REQUIRED_ROLES else ""
    choice = grid[i % 3].selectbox(
        f"{badge[sug.confidence]} {role.replace('_', ' ')}{required}", options,
        index=index, key=f"p4_map_{src_key}_{role}_{loaded_json is not None}",
        help=f"{mapping.ROLE_HELP[role]} Suggestion: {sug.column or 'none'} "
             f"({sug.reason}).")
    user_map[role] = None if choice == ui.NONE_OPTION else choice

errors = mapping.validate_mapping(df_raw, user_map)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

value_maps = {}
freq_col = user_map.get("frequency")
if freq_col and not mapping.frequency_is_numeric(df_raw[freq_col]):
    st.markdown("**Frequency labels → purchases per year** (edit any blank or wrong value)")
    defaults = mapping.default_frequency_map(df_raw[freq_col])
    if saved_values.get("frequency"):
        defaults.update({k: v for k, v in saved_values["frequency"].items()
                         if k in defaults})
    table = pd.DataFrame({"label": list(defaults),
                          "purchases_per_year": [defaults[k] for k in defaults]})
    edited = st.data_editor(
        table, hide_index=True, disabled=["label"], key=f"p4_freq_{src_key}",
        column_config={"purchases_per_year": st.column_config.NumberColumn(
            min_value=0.0, step=1.0)})
    fmap = dict(zip(edited["label"], edited["purchases_per_year"]))
    blank = [k for k, v in fmap.items() if pd.isna(v)]
    if blank:
        st.warning("Fill in a number for: " + ", ".join(blank))
        st.stop()
    value_maps["frequency"] = {k: float(v) for k, v in fmap.items()}

dayfirst = False
if user_map.get("transaction_date"):
    dayfirst = st.checkbox("Dates are day-first (e.g. 03/04/2025 = 3 April)",
                           value=saved_dayfirst, key=f"p4_dayfirst_{src_key}")

st.download_button("Save this mapping (JSON)",
                   mapping.mapping_to_json(user_map, value_maps, dayfirst),
                   file_name="column_mapping.json", mime="application/json",
                   key="p4_save_map")

try:
    mapped = mapping.apply_mapping(df_raw, user_map, value_maps, dayfirst)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
with st.expander("What the mapper did", expanded=False):
    for line in mapped.report:
        st.markdown(f"- {line}")
if mapped.rows_out == 0:
    st.error("No usable rows left after mapping.")
    st.stop()

# ---------------------------------------------------------------------------
# 3. Segmentation
# ---------------------------------------------------------------------------
st.header("3 · Segment your customers")
artifact = ui.load_artifact() if ui.artifacts_ready() else None
use_rfm = rfm.can_build_rfm(mapped.schema)
rfm_result = None

if use_rfm:
    st.info("Your file has many rows per customer, so we first build one row per "
            "customer: **Recency** (days since last purchase), **Frequency** "
            "(number of orders) and **Monetary** (total spend).")
    c1, c2 = st.columns(2)
    exclude = c1.checkbox("Exclude refunds/returns (amount ≤ 0)", value=True,
                          key="p4_refunds")
    use_log = c2.checkbox("Log-transform frequency and monetary (recommended: "
                          "they are right-skewed)", value=True, key="p4_log")
    try:
        rfm_result = rfm.build_rfm(mapped.df, exclude_nonpositive=exclude)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    for line in rfm_result.report:
        st.caption(line)
    for w in rfm_result.warnings:
        st.warning(w)
    customers = rfm_result.df
    mode = "B"
    st.caption("Mode A (apply the trained retail model) is not offered here: that "
               "model was trained on different features (purchase rate, spend, "
               "discount), not on recency/frequency/monetary.")
    features = list(rfm.RFM_FEATURES)
    log_features = list(rfm.RFM_LOG_FEATURES) if use_log else []
else:
    customers = mapped.df
    mode_a_ok = artifact is not None and all(
        f in customers.columns and customers[f].notna().any()
        for f in artifact["feature_names"])
    modes = ["B · Fit on my data (learn segments from this file)"]
    if mode_a_ok:
        modes.insert(0, "A · Apply trained model (assign to the retail segments)")
    mode = st.radio("Mode", modes, key="p4_mode")[0]
    if not mode_a_ok:
        st.caption("Mode A needs these mapped features: "
                   + (", ".join(artifact["feature_names"]) if artifact else
                      "a trained model") + ".")
    if mode == "B":
        numeric = [c for c in customers.columns
                   if pd.api.types.is_numeric_dtype(customers[c])
                   and c not in ("segment_id", "purchase_frequency_days")]
        default_feats, _ = fit.default_features(customers)
        features = st.multiselect("Features to cluster on", numeric,
                                  default=[f for f in default_feats if f in numeric],
                                  key=f"p4_feats_{src_key}")
        log_features = []

settings = json.dumps({"src": src_key, "map": user_map, "vm": value_maps,
                       "df": dayfirst, "mode": mode,
                       "feats": features if mode == "B" else None,
                       "log": log_features if mode == "B" else None,
                       "rfm": use_rfm,
                       "ex": st.session_state.get("p4_refunds")}, sort_keys=True,
                      default=str)
settings_key = hashlib.md5(settings.encode()).hexdigest()

if st.button("Run segmentation", type="primary", key="p4_run"):
    with st.spinner("Segmenting…"):
        try:
            if mode == "A":
                complete = customers.dropna(subset=artifact["feature_names"])
                seg = segmentation.predict_features(complete, artifact)
                X_new = seg[artifact["feature_names"]]
                result = {"mode": "A", "data": seg,
                          "dropped": len(customers) - len(complete),
                          "drift": segmentation.out_of_distribution_warnings(
                              X_new, artifact["training_stats"])}
            else:
                res = fit.fit_segmentation(customers, features=features,
                                           log_features=log_features)
                result = {"mode": "B", "fit": res, "data": res.data}
            st.session_state["p4_result"] = {"key": settings_key, **result}
        except ValueError as exc:
            st.session_state.pop("p4_result", None)
            st.error(str(exc))

result = st.session_state.get("p4_result")
if result and result["key"] != settings_key:
    st.caption("Settings changed — press **Run segmentation** again.")
    result = None

segmented = None
if result:
    segmented = result["data"]
    profile = segmentation.segment_profile(segmented)
    profile["description"] = profile.apply(segmentation.describe_segment, axis=1)
    if result["mode"] == "A":
        st.info("The model is frozen. Your rows were assigned to the existing "
                "retail segments; they were not re-clustered.")
        if result["dropped"]:
            st.warning(f"{result['dropped']:,} row(s) without all model features "
                       "were left out.")
        if result["drift"]:
            st.warning("This file looks different from the training data, so it is "
                       "being force-fitted into the existing segments. Consider "
                       "Mode B.\n\n" + "\n".join(f"- {w}" for w in result["drift"]))
        ui.segment_distribution(segmented["segment"], key="p4_dist")
    else:
        res = result["fit"]
        for w in res.warnings:
            st.warning(w)
        st.success(f"Chosen k = {res.k} · silhouette = {res.silhouette:.3f} · "
                   f"features: {', '.join(res.features)}"
                   + (f" (log: {', '.join(res.log_features)})" if res.log_features
                      else ""))
        st.caption(f"Why this k: {res.reason}")
        ui.sweep_charts(res.sweep, res.k, key="p4_sweep")
        ui.pca_chart(res.pca, res.pca_variance, key="p4_pca")
    st.subheader("Segment profile")
    ui.profile_table(profile)

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download segmented data (CSV)",
                       segmented.to_csv(index=False).encode("utf-8"),
                       file_name="segmented_data.csv", mime="text/csv",
                       key="p4_dl_csv")
    if result["mode"] == "B":
        c2.download_button("Download fitted model (joblib)", res.model_bytes(),
                           file_name="fit_result.joblib", key="p4_dl_model")
        c3.download_button("Download metrics (JSON)", res.metrics_json(),
                           file_name="metrics.json", mime="application/json",
                           key="p4_dl_metrics")

# ---------------------------------------------------------------------------
# 4. Dashboard
# ---------------------------------------------------------------------------
st.header("4 · Dashboard")
dash_df = mapped.df
if segmented is not None:
    if use_rfm:
        dash_df = mapped.df.merge(segmented[["customer_id", "segment"]],
                                  on="customer_id", how="left")
    else:
        dash_df = segmented
else:
    st.caption("Run a segmentation to add segment charts.")
schema = {role: col for role, col in mapped.schema.items()}
filtered = ui.sidebar_filters(dash_df, schema, key=f"p4_{src_key}")
st.caption(f"Showing {len(filtered):,} of {len(dash_df):,} rows.")
ui.render_dashboard(de.build_dashboard(filtered, schema), key="p4_dash")
ui.footer()
