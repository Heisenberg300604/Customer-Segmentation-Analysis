# Customer Analytics Platform — K-Means Segmentation + Streamlit

A retail analytics project that grew in two stages:

1. **Baseline:** Python (pandas) → PostgreSQL → 10 SQL business queries → Power BI dashboard,
   on 3,900 retail purchases.
2. **This extension:** the hand-written *New / Returning / Loyal* rule is replaced with an
   **unsupervised K-Means segmentation** chosen by a written, testable rule. It's wrapped in a
   **Streamlit web app** where anyone can upload their own customer file (even with
   different column names, or raw transactions with dates) and get segments plus a dashboard back.

> The reusable deliverable is the **pipeline**, not the specific segments. Every number shown in
> the app or in this README is computed from the data by the code.

---

## 1. Run it in one command

You need **Python 3.11 or 3.12** ([python.org/downloads](https://www.python.org/downloads/)) and git.

**macOS / Linux**

```bash
git clone https://github.com/Heisenberg300604/Customer-Segmentation-Analysis.git
cd Customer-Segmentation-Analysis
./run.sh
```

**Windows** (Command Prompt)

```bat
git clone https://github.com/Heisenberg300604/Customer-Segmentation-Analysis.git
cd Customer-Segmentation-Analysis
run.bat
```

The first run creates a virtual environment, installs the packages (1–2 minutes), and opens the
app at **http://localhost:8501**. Later runs start immediately. The trained model is already in
the repo, so no training is needed.

| What you want | macOS / Linux | Windows |
|---|---|---|
| Open the web app | `./run.sh` | `run.bat` |
| Run all tests | `./run.sh test` | `run.bat test` |
| Retrain the model (also regenerates figures + docs) | `./run.sh train` | `run.bat train` |

<details>
<summary>Manual setup (if you prefer typing the steps yourself)</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train                  # optional: artifacts are already committed
pytest                               # 61 tests
streamlit run app/app.py
```

Other commands:

```bash
python -m src.make_docs                        # rebuild docs from artifacts/metrics.json
python -m tests.make_synthetic_transactions    # rebuild data/sample_transactions.csv
pip install -r requirements-db.txt             # optional: PostgreSQL loader
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/dbname"
python -m src.load_to_postgres                 # then run sql/02_business_queries.sql
```
</details>

---

## 2. What the app does

| Page | What it shows |
|---|---|
| **Overview Dashboard** | The Power BI dashboard rebuilt on the web: KPI cards, revenue/sales by category and age group, subscription donut, gender, top products, shipping. Also new K-Means segment charts, a segment profile in plain English, and a *Model validation* panel (sweep table, elbow, silhouette, PCA, surrogate tree). Sidebar filters update everything. |
| **Upload & Predict** | Upload a CSV/Excel file with the retail columns (or tick *Use sample dataset*). Each row is assigned to the existing segments by the **frozen** model, with a warning if the file looks different from the training data. Download the results. |
| **Segment Explorer** | Filter by segment, gender, category, age group or subscription, and search by Customer ID. Compare the filtered group with everyone and download it. |
| **Analyze Your Data** | Any file: **map columns** (auto-suggested, with confidence badges) → **RFM** if it has customer + date + amount → **Mode A** (apply the trained model) or **Mode B** (*fit on my data*: same method, new segments) → full dashboard. |

---

## 3. How it works

```mermaid
flowchart LR
    A[Raw CSV] --> B[Clean + engineer<br/>src/pipeline.py]
    B --> C[(PostgreSQL)] --> D[SQL queries] --> E[Power BI]
    B --> F[K-Means + selection rule<br/>src/train.py] --> G[artifacts/<br/>model, CSV, metrics, figures]
    G --> H[Streamlit app<br/>app/app.py]
    U[Any uploaded file] --> M[Column mapper<br/>src/mapping.py] --> R[RFM<br/>src/rfm.py] --> FIT[Fit on my data<br/>src/fit.py] --> H
```

One cleaning module (`src/pipeline.py`) is shared by training and the app, so an uploaded
row is transformed exactly like a training row.

**The ML steps, in plain English**

1. **Clean:** column names → snake_case, missing review ratings filled with the median of the
   same category, `promo_code_used` dropped (it is identical to `discount_applied`).
2. **Engineer features:** frequency label → purchases per year (Weekly = 52 … Annually = 1),
   discount → 0/1, age → four age-quartile groups.
3. **Scale:** `StandardScaler` puts every feature on the same scale (mean 0, std 1).
4. **Sweep k = 2…8:** for every k, fit K-Means and record *inertia* (elbow) and *silhouette*.
   This is done for the PRD's 4-feature design and three alternative feature sets.
5. **Choose with a written rule** (`src/config.py`): Kneedle elbow → quality gate
   (silhouette ≥ 0.25, every segment ≥ 5%) → ignore all-discrete feature sets → highest silhouette.
6. **Label automatically:** each cluster is named from how its average differs from the overall
   average (e.g. far above-average purchase rate → "High-Frequency"). Nothing is named by hand.
7. **Explain:** a PCA plot shows the clusters in 2-D. A depth-3 decision tree (a *surrogate*)
   shows which features separate them.

---

## 4. Results

<!-- RESULTS:START -->
<!-- This block is generated by `python -m src.make_docs` from artifacts/metrics.json. Do not edit by hand. -->

**Chosen model:** `rate_amt_disc` = purchase_rate_per_year, purchase_amount, discount_flag · **k = 5** ·
**silhouette = 0.433** · trained on 3,900 rows.

**How k was picked (say this in the viva):** We fixed a rule in `config.py` and applied it to every feature set: take the elbow of the inertia curve (Kneedle method), accept it only if silhouette ≥ 0.25 and every segment holds ≥ 5% of customers, ignore feature sets made only of discrete values (their silhouette is inflated), and keep the highest silhouette. That selected **purchase_rate_per_year, purchase_amount, discount_flag** with **k = 5** (silhouette **0.433**).

**Silhouette for every feature set and k** (bold = chosen, ◀ = elbow):

| k | `prd_4` | `rate_disc` | `rate_amt_disc` | `rate_prev_disc` |
|---|---|---|---|---|
| 2 | 0.272 | 0.574 | 0.368 | 0.369 |
| 3 | 0.235 | 0.675 | 0.408 | 0.409 |
| 4 | 0.267 ◀ elbow | 0.737 ◀ elbow | 0.406 | 0.403 |
| 5 | 0.274 | 0.808 | **0.433** ◀ elbow | 0.426 ◀ elbow |
| 6 | 0.274 | 0.870 | 0.462 | 0.455 |
| 7 | 0.281 | 0.908 | 0.459 | 0.453 |
| 8 | 0.282 | 0.937 | 0.460 | 0.464 |

**Selection rule applied to each feature set:**

| Feature set | Features | Elbow k | Silhouette at elbow | Smallest segment | Eligible | Passes gate | Segment sizes at k=3 |
|---|---|---|---|---|---|---|---|
| `prd_4` | purchase_rate_per_year, purchase_amount, previous_purchases, discount_flag | 4 | 0.267 | 13.8% | yes | yes | Discount-Driven 1,677 / Full-Price · Budget 1,114 / Big-Spender · Full-Price 1,109 |
| `rate_disc` | purchase_rate_per_year, discount_flag | 4 | 0.737 | 18.0% | no (all discrete) | yes | Full-Price Loyal 1,918 / Discount-Driven Regular 1,443 / High-Frequency Shoppers 539 |
| `rate_amt_disc` | purchase_rate_per_year, purchase_amount, discount_flag | 5 | 0.433 | 13.8% | yes | yes | Full-Price Loyal 1,918 / Discount-Driven Regular 1,443 / High-Frequency Shoppers 539 |
| `rate_prev_disc` | purchase_rate_per_year, previous_purchases, discount_flag | 5 | 0.426 | 13.8% | yes | yes | Full-Price Loyal 1,918 / Discount-Driven Regular 1,443 / High-Frequency Shoppers 539 |

**Segment profile (final model):**

| Segment | Customers | % of base | Purchases/yr | Avg spend | Discount use | Subscribers | Plain English |
|---|---|---|---|---|---|---|---|
| Full-Price · Budget | 967 | 24.8% | 12.1 | $39.68 | 0% | 0% | Buys ~12×/yr, 0% use a discount, avg spend $39.68, 25 previous purchases, mostly Clothing |
| Big-Spender · Full-Price | 951 | 24.4% | 11.8 | $81.12 | 0% | 0% | Buys ~12×/yr, 0% use a discount, avg spend $81.12, 25 previous purchases, mostly Clothing |
| Discount-Driven · Big-Spender | 731 | 18.7% | 12.4 | $79.73 | 100% | 62% | Buys ~12×/yr, 100% use a discount, avg spend $79.73, 26 previous purchases, mostly Clothing |
| Discount-Driven · Budget | 712 | 18.3% | 11.5 | $38.63 | 100% | 62% | Buys ~12×/yr, 100% use a discount, avg spend $38.63, 26 previous purchases, mostly Clothing |
| High-Frequency | 539 | 13.8% | 52.0 | $58.97 | 43% | 29% | Buys ~52×/yr, 43% use a discount, avg spend $58.97, 26 previous purchases, mostly Clothing |

**Surrogate explanation:** a depth-3 decision tree imitates the
clusters with 99.5% agreement; feature importances:
discount_flag 0.46, purchase_rate_per_year 0.30, purchase_amount 0.24.
<!-- RESULTS:END -->

The PRD quoted numbers that did not reproduce; see **[docs/PRD_reconciliation.md](docs/PRD_reconciliation.md)**.
The short answers for the defense are in **[docs/DEFENSE_CHEAT_SHEET.md](docs/DEFENSE_CHEAT_SHEET.md)**.

---

## 5. How it generalizes, and its honest limits

**Generalizes**
* **Column mapper:** different column names (`order_total`, `cust_id`, `sex` …) are matched
  automatically. You can correct any guess and save the mapping as JSON.
* **Mode B, fit on my data:** the same sweep, rule and auto-labelling run on *your* file, so
  nothing is forced into the retail segments.
* **RFM:** transaction files (many rows per customer, with dates) become one row per customer
  (Recency, Frequency, Monetary) before clustering.
* **Adaptive dashboard:** only the charts your columns support are shown; the rest are listed.

**Limits**
* The frozen model (page 2 / Mode A) force-fits foreign data into the retail segments. The app
  warns about this.
* Only a few behavioural features are available (no real dates or baskets in the retail file).
* K-Means assumes roughly round, similar-sized clusters. A low silhouette means weak segments.
* The segments found are specific to each dataset; only the method generalizes.
* One file at a time (no joins), and at least 2 numeric features are needed.
* Streamlit Community Cloud is free, best-effort hosting: fine for a demo, not for production.

---

## 6. Viva / interview notes

**Project basics**

* **What problem does it solve?** It groups customers by behaviour so a retailer can target
  them differently (e.g. discounts for discount-driven buyers, loyalty perks for frequent ones),
  and lets any retailer do the same with their own file.
* **What did the mentor criticise, and how did you fix it?** (1) The segmentation was a fixed
  rule tied to one dataset, so we replaced it with K-Means and a written selection rule.
  (2) Nobody else could use it, so we built a web app with upload, column mapping and
  "fit on my data".
* **What is the dataset?** 3,900 retail purchases, 18 columns: demographics, product, amount,
  rating, subscription, shipping, discount, previous purchases and purchase frequency.

**Data preparation**

* **Why median imputation by category?** The median is robust to outliers, and ratings may
  differ by category, so the category median is a closer guess than the global one.
* **Why drop `promo_code_used`?** It was identical to `discount_applied` on every row. Keeping
  both would double-count one signal.
* **Why `qcut` for age groups?** It makes four equal-sized age quartiles, which reproduces the
  baseline SQL numbers. For uploads, the *training* quartile edges are reused so groups stay
  comparable.

**K-Means**

* **How does K-Means work?** Pick k centres, assign every point to its nearest centre, move each
  centre to the mean of its points, and repeat until nothing changes. It minimises
  *inertia* (the sum of squared distances to the centres).
* **Why scale first?** K-Means uses distances. Unscaled, spend (tens of dollars) would swamp a
  0/1 discount flag.
* **What are `n_init=10` and `random_state=42`?** K-Means depends on its starting centres, so it
  runs 10 starts and keeps the best one. The fixed seed makes results identical on every run.
* **Elbow method?** Plot inertia against k and look for the bend where extra clusters stop
  helping much. We find the bend automatically with the Kneedle method (largest gap below the
  line joining the first and last points).
* **Silhouette score?** For each point: (b − a) / max(a, b), where a is the average distance
  to its own cluster and b is the distance to the nearest other cluster. It ranges from −1 to 1;
  higher means tighter, better-separated clusters.
* **Why not just take the highest silhouette?** With discrete features (a 0/1 flag and 7 frequency
  levels) silhouette keeps rising as k grows, because clusters collapse onto a few points. So we
  use a written rule and exclude all-discrete feature sets.
* **Why K-Means and not DBSCAN or hierarchical clustering?** It is simple, fast, easy to explain
  (a segment is its average customer) and can label new customers instantly by nearest centre.
  DBSCAN needs density tuning and can leave points unassigned. Hierarchical clustering is slow
  on large files and has no simple predict step.
* **What does PCA do here?** It squeezes the scaled features into 2 dimensions only to *draw*
  the clusters. It is not used for clustering.
* **What is the surrogate tree?** K-Means has no feature importance, so a small decision tree is
  trained to imitate the cluster labels. Its rules and importances explain the segments. Its
  "fidelity" is how often it agrees with K-Means.

**Honesty and validation**

* **Did you reproduce the PRD?** Not fully: the PRD's 0.292 silhouette at k=3 did not
  reproduce. The measured values are documented and shown instead (see Results above and
  `docs/PRD_reconciliation.md`).
* **How do you know `predict()` matches training?** A test re-predicts all 3,900 rows and
  requires 100% agreement with the saved segments. Another test retrains and requires identical
  labels.
* **Bi-Weekly: 26 or 104 per year?** It's ambiguous. We default to 26 and report the effect of
  104 in the reconciliation doc.

**RFM and reusability**

* **What is RFM?** Recency (days since the last purchase), Frequency (number of orders) and
  Monetary (total spend), computed per customer from transactions.
* **Why `log1p` on frequency and monetary?** Both are right-skewed: a few customers buy far more.
  The log compresses that tail so a handful of big spenders don't dominate the clusters.
* **Mode A vs Mode B?** Mode A applies the frozen retail model, so new data is forced into the
  existing segments. Mode B re-runs the full method on the uploaded file.
* **How does the column mapper work?** Synonym lists, then fuzzy matching (`difflib`), then a
  data-type check (numeric, dates, gender values). The user confirms every role.

**Engineering**

* **Testing:** 61 pytest tests: pipeline baselines, segmentation, mapper, RFM (hand-calculated),
  fit, dashboard, and headless Streamlit page tests (`AppTest`).
* **Why Streamlit?** Pure Python, no separate frontend, free hosting, and caching for speed.
* **Deployment:** push to GitHub, then Streamlit Community Cloud → *New app* → main file
  `app/app.py`.
* **What would you improve?** Real timestamps and basket data, Gaussian Mixture Models for
  non-round clusters, segment stability checks (bootstrap), and scheduled retraining.

---

## 7. Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub (done if you cloned it from there).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → choose the repo,
   branch `main`, main file **`app/app.py`**.
3. Under *Advanced settings* pick **Python 3.11** → **Deploy**.

The app reads the committed artifacts, so no database, server or secrets are needed.

---

## 8. Project structure

```
├── app/
│   ├── app.py                     # Streamlit entry (navigation)
│   ├── ui.py                      # theme, KPI cards, upload reader, filters, charts
│   └── pages/                     # 1 Overview · 2 Upload & Predict · 3 Explorer · 4 Analyze Your Data
├── src/
│   ├── config.py                  # every constant + the k-selection rule
│   ├── pipeline.py                # shared cleaning + feature engineering
│   ├── segmentation.py            # sweep, elbow, rule, auto-labels, profile, predict
│   ├── train.py                   # trains the retail model, writes artifacts + docs
│   ├── mapping.py                 # column mapper
│   ├── rfm.py                     # transactions → recency/frequency/monetary
│   ├── fit.py                     # "fit on my data" for any file
│   ├── dashboard_engine.py        # one dashboard for any schema
│   ├── make_docs.py               # writes the number-carrying docs from metrics.json
│   └── load_to_postgres.py        # optional PostgreSQL loader
├── artifacts/                     # model, clustered CSV, metrics.json, figures/
├── data/                          # retail dataset, sample upload, synthetic transactions
├── sql/                           # table DDL + 10 business queries + segment query
├── docs/                          # PRD reconciliation, defense cheat sheet
├── tests/                         # 61 tests + synthetic data generator
├── run.sh / run.bat               # one-command setup and run
└── requirements.txt               # pinned (Python 3.11)
```

---

## 9. Troubleshooting

* **`Python 3.11 or 3.12 was not found`:** install it from python.org, then run the script again.
  Python 3.13 is not supported by the pinned NumPy version.
* **Port 8501 already in use:** `.venv/bin/python -m streamlit run app/app.py --server.port 8502`.
* **"The trained model was not found":** run `./run.sh train` (or `python -m src.train`).
