"""Page 2 - Upload & Predict: assign an uploaded file to the EXISTING segments
with the frozen, already-trained model. It never re-clusters."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st  # noqa: E402
import ui  # noqa: E402

from src import config, pipeline, segmentation  # noqa: E402

ui.setup("Upload & Predict",
         "Assign your customers to the trained segments")
ui.require_artifacts()
artifact = ui.load_artifact()

st.info("The model is frozen. Uploaded data is assigned to the existing segments; "
        "it is never re-clustered. To learn new segments from your own data, use "
        "**Analyze Your Data**.")

st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in
                                                  config.REQUIRED_COLUMNS_MODEL))
use_sample = st.checkbox("Use sample dataset", key="p2_sample",
                         help="Loads data/sample_upload.csv so the page works "
                              "with zero setup.")
uploaded = None if use_sample else st.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xlsx"], key="p2_upload")

if use_sample:
    df_raw = ui.load_csv(str(config.SAMPLE_UPLOAD_PATH))
    source = "sample_upload.csv"
elif uploaded is not None:
    df_raw, error = ui.read_upload(uploaded)
    if error:
        st.error(error)
        st.stop()
    source = uploaded.name
else:
    st.caption("Upload a file or tick **Use sample dataset** to begin.")
    ui.footer()
    st.stop()

ok, missing_model, missing_dash = pipeline.validate_columns(df_raw)
if not ok:
    st.error("Missing required column(s): " + ", ".join(missing_model)
             + ". Add them, or use **Analyze Your Data** to map differently "
               "named columns.")
    st.stop()
if missing_dash:
    st.warning("Optional column(s) not found (some details will be missing): "
               + ", ".join(missing_dash))

st.subheader(f"Preview · {source} · {len(df_raw):,} rows")
st.dataframe(df_raw.head(20), use_container_width=True)

try:
    result = segmentation.predict(df_raw, artifact)
except ValueError as exc:
    st.error(f"Could not process the file: {exc}")
    st.stop()

X_new = pipeline.build_features(result, artifact["feature_names"])
drift = segmentation.out_of_distribution_warnings(X_new, artifact["training_stats"])
if drift:
    st.warning("This file looks different from the training data, so it is being "
               "force-fitted into the existing segments:\n\n"
               + "\n".join(f"- {w}" for w in drift))

st.subheader("Segment distribution")
ui.segment_distribution(result["segment"], key="p2_dist")

st.subheader("Segment profile of this file")
profile = segmentation.segment_profile(result)
profile["description"] = profile.apply(segmentation.describe_segment, axis=1)
ui.profile_table(profile)

st.download_button("Download results (CSV)",
                   result.to_csv(index=False).encode("utf-8"),
                   file_name="segmented_customers.csv", mime="text/csv",
                   key="p2_download")
ui.footer()
