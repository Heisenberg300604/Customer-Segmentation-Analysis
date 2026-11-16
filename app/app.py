"""Streamlit entry point. Run from the project folder:

    streamlit run app/app.py

Streamlit Community Cloud: set the main file to app/app.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Customer Analytics Platform", page_icon="📊",
                   layout="wide")

PAGES = Path(__file__).resolve().parent / "pages"
nav = st.navigation([
    st.Page(PAGES / "1_Overview_Dashboard.py", title="Overview Dashboard",
            icon="📊", default=True),
    st.Page(PAGES / "2_Upload_and_Predict.py", title="Upload & Predict", icon="📤"),
    st.Page(PAGES / "3_Segment_Explorer.py", title="Segment Explorer", icon="🔎"),
    st.Page(PAGES / "4_Analyze_Your_Data.py", title="Analyze Your Data", icon="🧭"),
])
nav.run()
