"""Project-wide constants.

Every column name, mapping, seed and path lives here so that the training
script and the Streamlit app always agree. Nothing in this file is a *result*:
metrics and segment sizes are computed by the code and saved to artifacts/.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repo root, so they work locally and on Streamlit Cloud)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "Customer_dataset.csv"
SAMPLE_UPLOAD_PATH = DATA_DIR / "sample_upload.csv"
CLUSTERED_CSV_PATH = ARTIFACTS_DIR / "clustered_customers.csv"
MODEL_PATH = ARTIFACTS_DIR / "kmeans_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
# Raw column names exactly as they appear in Customer_dataset.csv.
RAW_COLUMNS = [
    "Customer ID", "Age", "Gender", "Item Purchased", "Category",
    "Purchase Amount (USD)", "Location", "Size", "Color", "Season",
    "Review Rating", "Subscription Status", "Shipping Type",
    "Discount Applied", "Promo Code Used", "Previous Purchases",
    "Payment Method", "Frequency of Purchases",
]

# After clean() every column is snake_case. A few snake_case names are long or
# awkward, so we shorten them to the names used everywhere else in the code.
COLUMN_ALIASES = {
    "purchase_amount_usd": "purchase_amount",
}

# Columns the K-Means model cannot run without (raw names, for error messages).
REQUIRED_COLUMNS_MODEL = [
    "Frequency of Purchases",
    "Purchase Amount (USD)",
    "Previous Purchases",
    "Discount Applied",
]

# Columns the dashboard uses. If one is missing the app shows a warning and
# hides the affected charts, but segmentation still works.
REQUIRED_COLUMNS_DASHBOARD = [
    "Customer ID", "Age", "Gender", "Category", "Item Purchased",
    "Review Rating", "Subscription Status", "Shipping Type", "Location",
]

# Columns whose values are Yes/No text. clean() normalises them to "Yes"/"No".
YES_NO_COLUMNS = ["discount_applied", "promo_code_used", "subscription_status"]

# Column dropped in cleaning: it is perfectly redundant with discount_applied.
REDUNDANT_COLUMNS = ["promo_code_used"]

# ---------------------------------------------------------------------------
# Frequency label -> purchases per year
# ---------------------------------------------------------------------------
# "Bi-Weekly" is ambiguous: it can mean "every two weeks" (26/yr, same as
# Fortnightly) or "twice a week" (104/yr). We default to 26 because the dataset
# also has "Fortnightly" and "Weekly" and no other twice-a-week style label.
# Change this to 104 and re-run `python -m src.train` to see whether the
# segments change (the result is reported in docs/PRD_reconciliation.md).
BI_WEEKLY_PER_YEAR = 26

FREQUENCY_TO_PER_YEAR = {
    "Weekly": 52,
    "Bi-Weekly": BI_WEEKLY_PER_YEAR,
    "Fortnightly": 26,
    "Monthly": 12,
    "Quarterly": 4,
    "Every 3 Months": 4,
    "Annually": 1,
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Age groups are the four age quartiles of the training data, youngest first.
# This reproduces the SQL/Power BI revenue-by-age-group numbers exactly.
AGE_GROUP_LABELS = ["Young Adult", "Adult", "Middle-aged", "Senior"]

# The model's feature matrix, always in this order (the PRD's 4-feature design).
MODEL_FEATURES = [
    "purchase_rate_per_year",
    "purchase_amount",
    "previous_purchases",
    "discount_flag",
]

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE = range(2, 9)  # k = 2..8 inclusive
KMEANS_N_INIT = 10

# ---------------------------------------------------------------------------
# Feature-set comparison and the k / feature-set selection rule (see README)
# ---------------------------------------------------------------------------
# The PRD's 4-feature design is the primary experiment; the others are the
# documented alternatives. Every set is run through the same k-sweep.
PRIMARY_FEATURE_SET = "prd_4"
FEATURE_SETS = {
    "prd_4": ["purchase_rate_per_year", "purchase_amount",
              "previous_purchases", "discount_flag"],
    "rate_disc": ["purchase_rate_per_year", "discount_flag"],
    "rate_amt_disc": ["purchase_rate_per_year", "purchase_amount",
                      "discount_flag"],
    "rate_prev_disc": ["purchase_rate_per_year", "previous_purchases",
                       "discount_flag"],
}

# SELECTION RULE (written before looking at which set it would pick):
#  1. For each feature set, k = elbow of the inertia curve over K_RANGE, found
#     with the Kneedle method (largest gap below the straight line joining the
#     first and last points of the normalised curve). No tuning parameter.
#  2. Quality gate: that k is accepted only if its silhouette >= MIN_SILHOUETTE
#     and its smallest cluster holds >= MIN_CLUSTER_SHARE of customers.
#  3. Eligibility: a feature set must contain at least one continuous feature
#     (>= CONTINUOUS_MIN_UNIQUE distinct values). Sets made only of discrete
#     features are reported but not eligible: their rows collapse onto a
#     handful of distinct points, so silhouette rises towards 1 as k grows
#     and stops measuring real cluster quality.
#  4. Among eligible sets that pass the gate, choose the highest silhouette at
#     its elbow k (ties within SILHOUETTE_TIE -> fewer features).
#  5. If none pass, fall back to the primary set at its max-silhouette k and
#     flag the result as weak.
MIN_SILHOUETTE = 0.25          # common rule-of-thumb for "weak structure"
MIN_CLUSTER_SHARE = 0.05       # a segment under 5% is too small to act on
CONTINUOUS_MIN_UNIQUE = 20
SILHOUETTE_TIE = 0.005

# Auto-labelling: a cluster "has" a trait when its mean differs from the
# overall mean by at least this many standard deviations.
TRAIT_Z = 0.5

# Surrogate explanation: a shallow decision tree trained on cluster labels.
SURROGATE_MAX_DEPTH = 3
