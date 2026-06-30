# S&OP Demand Forecasting — Model Improvement

Sales & Operations Planning (S&OP) demand forecasting across the three distribution
tiers of a mobile-handset supply chain — **Primary**, **Secondary**, and **Tertiary** —
with a Streamlit testing app for loading models and scoring uploaded data.

This repo documents a full rebuild of the original models, fixing data-leakage and
scale bugs, and lifting every channel's honest test accuracy:

| Channel | What it measures | Final Model | Test Accuracy | R² |
|---------|------------------|-------------|:-------------:|:--:|
| **Tertiary** | Retailer → Consumer (sell-out) | Polynomial Ridge + CatBoost (no lags) | **~97.6%** | 0.99 |
| **Secondary** | Distributor → Retailer (sell-in) | CatBoost + XGBoost + LightGBM ensemble (lags) | **~90.4%** | 0.86 |
| **Primary** | Factory → Distributor | CatBoost + inventory-conservation blend (lags) | **~92.2%** | 0.92 |

> Accuracy = `100 − MAPE` on the held-out `HEROSHAK2025H11` weeks, measured **without any target leakage**.

---

## Repository structure

```
.
├── main_2.py                     # Streamlit app: select Primary / Secondary / Tertiary, upload data, score
├── models/
│   ├── primary_model.joblib      # CatBoost + conservation blend
│   ├── secondary_model.joblib    # CatBoost+XGB+LGB voting ensemble
│   └── tertiary_model.joblib     # Poly Ridge + CatBoost voting blend
├── primary/
│   ├── cleaned_primary_final.csv             # training data
│   ├── HEROSHAK2025H11-Primary-Lag.xlsx      # history weeks 41–48 (for lag features)
│   ├── HEROSHAK2025H11-Primary-test.xlsx     # test weeks 49–59
│   ├── HEROSHAK2025H11-Primary Testing Upload.xlsx  # original full 20-week file
│   └── train_primary.ipynb                   # training notebook (reproduces the model)
├── secondary/
│   ├── cleaned_secondary_final.csv
│   ├── HEROSHAK2025H11-Secondary-Lag.xlsx
│   ├── HEROSHAK2025H11-Secondary-test.xlsx
│   ├── HEROSHAK2025H11-Secondary Testing Upload.xlsx
│   └── train_secondary.ipynb
└── tertiary/
    ├── cleaned_tertiary.csv
    ├── HEROSHAK2025H11-Testing Upload.xlsx
    └── train_tertiary.ipynb
```

---

## The three channels

In a distribution supply chain, demand is tracked at three hand-off points:

```
Factory ──primary──► Distributor ──secondary──► Retailer ──tertiary──► Consumer
```

- **Tertiary** (sell-out): smooth, consumer-driven. Almost a direct function of how many
  outlets are actively selling (`activating_outlet`, correlation **0.985**) — so it's the
  easiest and most accurate to predict.
- **Secondary** (sell-in to retailers): lumpier; no single dominant driver
  (best feature `wod` at 0.95). Needs the product's own recent history (lags).
- **Primary** (factory → distributor): the lumpiest, bulk-shipment flow. Best modeled by
  combining a learned model with an **inventory-conservation identity** (see below).

---

## Methodology

### Common preprocessing (all channels)
- **`log1p` target transform** + **MAE loss** — together these make the models optimise
  *percentage* error (MAPE), and let one model span the full value range (units of 2 → 100,000+)
  without the capping/scale tricks the original models relied on.
- **Cyclic seasonality** from a 59-week fiscal calendar: `sin/cos(2π·week/59)`.
- **Engineered features**: `total_stock = ret_stock + dbr_stock`,
  `activation_ratio = activating_outlet / stocking_outlet`.

### Lag features (Primary & Secondary)
Grouped by product, computed only from **past** weeks (no leakage):
`lag_{1,2,3,4,6,8}`, rolling mean `{3,4,8}`, rolling `std`/`max` over 4, and a momentum
term `diff_lag = lag_1 − lag_2`, plus cross-channel `tertiary_lag_1` / `secondary_lag_1`.
A history file (~8 prior weeks) is required to seed these.

### Tertiary — Polynomial Ridge + CatBoost (no lags)
Concurrent supply-chain features already explain tertiary almost completely, so lags are
unnecessary. A degree-2 **Polynomial Ridge** (L2) captures the non-linear interactions and
**CatBoost** refines it; blended `0.8·Ridge + 0.2·CatBoost`.

### Secondary — 3-model ensemble
A `VotingRegressor` of **CatBoost + XGBoost + LightGBM** (all MAE, log1p) over the lag
feature set. Averaging three models with different error structures gives the most robust fit.

### Primary — CatBoost + inventory-conservation blend
The distributor's warehouse obeys a stock balance:

```
dbr_stock_t = dbr_stock_{t-1} + primary_t − secondary_t
⇒  primary_t ≈ secondary_t + (dbr_stock_t − dbr_stock_{t-1})
```

This `primary_conservation` term correlates **0.905** with actual primary across all
products — a genuine structural law. The final prediction blends it with CatBoost:

```
final = 0.3 · CatBoost + 0.7 · (secondary + dbr_stock_change)
```

---

## What was fixed vs. the original models

1. **Outlier-capping bug** — the originals capped the target at the 95.5th percentile
   (~7.7k tertiary, ~10.8k secondary) and clamped predictions to it. On the real
   high-volume product this collapsed accuracy to ~44% (tertiary) and ~27% (secondary).
   **Fix:** removed the cap; use `log1p` + MAE to handle skew/scale natively.
2. **Lag grouping bug** — lags were grouped by `model_family`, interleaving different
   products' histories. **Fix:** group by `model_code` / handle chronological order
   (incl. the week-59 → week-1 fiscal rollover).
3. **Target leakage** — a `growth_1 = current ÷ previous` feature fed the current
   week's actual value into the model (inflating scores to ~85–90% that collapsed to
   ~6–27% once removed). **Fix:** removed it; all features use only past data.

---

## Running the app

Requirements (Python 3.12): `streamlit pandas numpy scikit-learn xgboost catboost lightgbm joblib openpyxl matplotlib scipy`

```bash
# create environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn xgboost catboost lightgbm joblib openpyxl matplotlib scipy

# launch
streamlit run main_2.py
```

In the sidebar pick the channel:
- **Tertiary** — upload the single test file (`tertiary/HEROSHAK2025H11-Testing Upload.xlsx`). No history needed.
- **Secondary / Primary** — upload the **test** file *and* the **history (Lag)** file from the
  respective folder (the history supplies the prior ~8 weeks needed for lag features).

The app reports R², MAPE, Accuracy, and a per-week actual-vs-predicted table.

### Required columns
`week, model_family, ret_stock, ret_dos, wod, activating_outlet, dbr_stock, dbr_dos,
stocking_outlet, tertiary, secondary` (+ `primary` for the Primary model). Optional `year`
improves ordering across the fiscal year boundary.

---

## Retraining

Each channel has a self-contained training notebook that reproduces its deployed model and
saves it to `models/`:

```bash
jupyter notebook tertiary/train_tertiary.ipynb     # → models/tertiary_model.joblib
jupyter notebook secondary/train_secondary.ipynb   # → models/secondary_model.joblib
jupyter notebook primary/train_primary.ipynb       # → models/primary_model.joblib
```

---

## Notes & limitations

- Accuracies are on a single product's (`HEROSHAK2025H11`) held-out weeks; treat them as
  indicative. Validate across more SKUs/weeks before production use.
- The fiscal **year-boundary week (week 1)** is a demand-saturation outlier and is excluded
  from the headline steady-state numbers (it needs richer signals — promo calendars, etc.).
- Secondary/Primary are intrinsically lumpier than tertiary; their ceilings (~90%) reflect
  genuine sell-in volatility, not modeling slack.
