import io
import math
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - keep app usable when xgboost import differs.
    xgb = None


st.set_page_config(page_title="Lag Feature Model Testing (History Required)", layout="wide")
st.title("XGBoost Lag-Feature Model Testing (History Required)")


MODELS_DIR = "./models"
HISTORY_WEEKS_REQUIRED = 8  # approx. 2 months at weekly level
WEEKS_PER_YEAR = 59
UNIT_FACTOR_CHOICES = [1, 10, 100, 1000]

SECONDARY_REQUIRED = [
    "week",
    "model_family",
    "ret_stock",
    "ret_dos",
    "wod",
    "activating_outlet",
    "dbr_stock",
    "dbr_dos",
    "stocking_outlet",
    "tertiary",
    "secondary",
]

TERTIARY_REQUIRED = [
    "week",
    "model_family",
    "ret_stock",
    "ret_dos",
    "wod",
    "activating_outlet",
    "dbr_stock",
    "dbr_dos",
    "stocking_outlet",
    "plc_factor",
    "tertiary",
]

PRIMARY_REQUIRED = [
    "week",
    "model_family",
    "ret_stock",
    "ret_dos",
    "wod",
    "activating_outlet",
    "dbr_stock",
    "dbr_dos",
    "stocking_outlet",
    "tertiary",
    "secondary",
    "primary",
]

MODEL_CONFIG = {
    "Primary": {
        "artifact_file": "primary_model.joblib",
        "target_col": "primary",
        "force_unit_factor": 1,   # CatBoost log1p + conservation blend -> scale-native
        "requires_lags": True,    # uses primary lags + history -> needs history file
        "required_columns": PRIMARY_REQUIRED,
    },
    "Secondary": {
        "artifact_file": "secondary_model.joblib",
        "target_col": "secondary",
        "force_unit_factor": 1,   # CatBoost/XGB/LGB voting ensemble, log1p -> scale-native
        "requires_lags": True,    # uses secondary lags + history -> needs history file
        "required_columns": SECONDARY_REQUIRED,
    },
    "Tertiary": {
        "artifact_file": "tertiary_model.joblib",
        "target_col": "tertiary",
        "force_unit_factor": 1,   # Poly Ridge + CatBoost, log1p -> scale-native
        "requires_lags": False,   # no-lags model; test file only, no history needed
        "required_columns": TERTIARY_REQUIRED,
    },
}


def load_artifact(model_label: str):
    cfg = MODEL_CONFIG[model_label]
    path = os.path.join(MODELS_DIR, cfg["artifact_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    artifact = joblib.load(path)
    model = artifact["model"]
    features = artifact["features"]
    log_feature_cols = artifact.get("log_feature_cols", [])
    target_col = artifact.get("target", cfg["target_col"])
    target_log1p = bool(artifact.get("target_log1p", True))
    scaler = artifact.get("scaler")
    return model, features, log_feature_cols, target_col, target_log1p, scaler, path, artifact


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    week_num = pd.to_numeric(out["week"], errors="coerce").fillna(0)
    angle = 2.0 * math.pi * (week_num / float(WEEKS_PER_YEAR))
    out["seasonality_1(sin)"] = np.sin(angle)
    out["seasonality_2(cos)"] = np.cos(angle)
    return out


def order_rows_for_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        keys = ["model_family"]
        if "model_code" in out.columns:
            keys.append("model_code")
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out["week"] = pd.to_numeric(out["week"], errors="coerce")
        sort_keys = keys + ["year", "week"]
        if "_source_rank" in out.columns:
            sort_keys = ["_source_rank"] + sort_keys
        return out.sort_values(sort_keys).reset_index(drop=True)

    out["_upload_order"] = np.arange(len(out))
    group_keys = ["model_family"]
    if "model_code" in out.columns:
        group_keys.append("model_code")
    blocks = []
    for _, g in out.groupby(group_keys, sort=False):
        g = g.sort_values("_upload_order")
        weeks = g["week"].to_numpy()
        seg = np.zeros(len(g), dtype=int)
        s = 0
        for i in range(1, len(weeks)):
            if weeks[i] < weeks[i - 1]:
                s += 1
            seg[i] = s
        g = g.assign(_wk_seg=seg)
        blocks.append(g)
    out = pd.concat(blocks, ignore_index=True)
    sort_keys = list(group_keys)
    if "_source_rank" in out.columns:
        sort_keys.append("_source_rank")
    sort_keys.extend(["_wk_seg", "week", "_upload_order"])
    return out.sort_values(sort_keys).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    sort_keys = ["model_family"]
    if "model_code" in out.columns:
        sort_keys.append("model_code")
    if "_source_rank" in out.columns:
        sort_keys.append("_source_rank")
    if "year" in out.columns:
        sort_keys.append("year")
    if "_wk_seg" in out.columns:
        sort_keys.append("_wk_seg")
    sort_keys.append("week")
    if "_upload_order" in out.columns:
        sort_keys.append("_upload_order")
    out = out.sort_values(sort_keys).reset_index(drop=True)

    group_cols = ["model_family"]
    g = out.groupby(group_cols, sort=False)[target_col]

    # Lags of the target (past values only -> no leakage)
    for lag in (1, 2, 3, 4, 6, 8):
        out[f"{target_col}_lag_{lag}"] = g.shift(lag)

    shifted = g.shift(1)
    for window in (3, 4, 8):
        out[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=window).mean()
    out[f"{target_col}_roll_std_4"] = shifted.rolling(window=4, min_periods=4).std()
    out[f"{target_col}_roll_max_4"] = shifted.rolling(window=4, min_periods=4).max()
    out[f"{target_col}_diff_lag"] = out[f"{target_col}_lag_1"] - out[f"{target_col}_lag_2"]

    # past cross-channel signals (sell-out / sell-in one week ago)
    if "tertiary" in out.columns:
        out["tertiary_lag_1"] = out.groupby(group_cols, sort=False)["tertiary"].shift(1)
    if "secondary" in out.columns:
        out["secondary_lag_1"] = out.groupby(group_cols, sort=False)["secondary"].shift(1)

    # concurrent engineered features
    out["total_stock"] = pd.to_numeric(out.get("ret_stock", 0), errors="coerce").fillna(0) + pd.to_numeric(out.get("dbr_stock", 0), errors="coerce").fillna(0)
    out["activation_ratio"] = pd.to_numeric(out.get("activating_outlet", 0), errors="coerce").fillna(0) / (pd.to_numeric(out.get("stocking_outlet", 0), errors="coerce").fillna(0) + 1e-5)

    # inventory-conservation features (for primary): distributor stock balance
    #   dbr_stock_t = dbr_stock_{t-1} + primary_t - secondary_t  =>  primary_t ~= secondary_t + (dbr_stock_t - dbr_stock_{t-1})
    if "dbr_stock" in out.columns:
        out["dbr_stock_lag_1"] = out.groupby(group_cols, sort=False)["dbr_stock"].shift(1)
        out["dbr_change"] = pd.to_numeric(out["dbr_stock"], errors="coerce") - out["dbr_stock_lag_1"]
    if "ret_stock" in out.columns:
        out["ret_stock_lag_1"] = out.groupby(group_cols, sort=False)["ret_stock"].shift(1)
        out["ret_change"] = pd.to_numeric(out["ret_stock"], errors="coerce") - out["ret_stock_lag_1"]
    if "secondary" in out.columns and "dbr_change" in out.columns:
        out["primary_conservation"] = pd.to_numeric(out["secondary"], errors="coerce").fillna(0) + out["dbr_change"].fillna(0)

    return out


def apply_log_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)
            out[col] = np.log1p(vals)
    return out


def suggest_unit_factor(df: pd.DataFrame, target_col: str) -> int:
    """
    Heuristic for model unit alignment.
    Artifacts in this project are often trained on scaled-down weekly volumes.
    """
    if target_col not in df.columns:
        return 1
    med = pd.to_numeric(df[target_col], errors="coerce").abs().median()
    if pd.isna(med):
        return 1
    if med >= 5000:
        return 1000
    if med >= 500:
        return 100
    if med >= 50:
        return 10
    return 1


def apply_unit_scaling(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor == 1:
        return df.copy()
    out = df.copy()
    scale_cols = [
        "ret_stock",
        "dbr_stock",
        "wod",
        "stocking_outlet",
        "activating_outlet",
        "tertiary",
        "secondary",
        "primary",
    ]
    for col in scale_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / float(factor)
    return out


def build_model_matrix(df: pd.DataFrame, trained_features: list[str], log_feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in trained_features:
        if col not in out.columns:
            out[col] = 0.0
    X = out[trained_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = apply_log_features(X, log_feature_cols)
    return X


def evaluate_model_on_df(
    df: pd.DataFrame,
    target_col: str,
    trained_features: list[str],
    log_feature_cols: list[str],
    model,
    scaler,
    target_log1p: bool,
    output_unit_factor: int,
):
    if df.empty:
        return {"rows": 0, "mape": float("inf"), "accuracy": 0.0, "status": "empty_df"}

    X = build_model_matrix(df, trained_features, log_feature_cols)
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=trained_features, index=X.index)

    pred_model_scale = model.predict(X)
    pred = np.expm1(pred_model_scale) if target_log1p else pred_model_scale
    pred = np.clip(pred, 0, None)

    y_true = pd.to_numeric(df[target_col], errors="coerce").fillna(0).to_numpy() * output_unit_factor
    y_pred = pred * output_unit_factor
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100)
    return {
        "rows": int(len(df)),
        "mape": mape,
        "accuracy": max(0.0, 100.0 - mape),
        "status": "ok",
    }


def calibrate_unit_factor(
    historical_df: pd.DataFrame,
    target_col: str,
    trained_features: list[str],
    log_feature_cols: list[str],
    model,
    scaler,
    target_log1p: bool,
):
    diagnostics = []

    for factor in UNIT_FACTOR_CHOICES:
        hist_scaled = apply_unit_scaling(historical_df, factor)
        hist_feat = add_seasonality(hist_scaled)
        hist_feat = add_lag_features(hist_feat, target_col)

        lag_cols = [c for c in hist_feat.columns if "lag_" in c or "roll_mean" in c]
        hist_feat = hist_feat.dropna(subset=lag_cols).reset_index(drop=True)
        if hist_feat.empty:
            diagnostics.append(
                {"factor": factor, "rows": 0, "mape": float("inf"), "accuracy": 0.0, "status": "no_rows_after_lags"}
            )
            continue

        eval_res = evaluate_model_on_df(
            df=hist_feat,
            target_col=target_col,
            trained_features=trained_features,
            log_feature_cols=log_feature_cols,
            model=model,
            scaler=scaler,
            target_log1p=target_log1p,
            output_unit_factor=factor,
        )
        diagnostics.append(
            {
                "factor": factor,
                "rows": eval_res["rows"],
                "mape": eval_res["mape"],
                "accuracy": eval_res["accuracy"],
                "status": eval_res["status"],
            }
        )

    best = min(diagnostics, key=lambda x: (x["mape"], -x["rows"]))
    return int(best["factor"]), diagnostics


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, on_bad_lines="skip")
    return pd.read_excel(uploaded_file)


def get_xgboost_compatibility_notes(model, artifact: dict):
    notes = []
    version_info = {
        "runtime_xgboost_version": getattr(xgb, "__version__", "unavailable"),
        "artifact_xgboost_version": artifact.get("xgboost_version", "not_stored"),
    }

    if xgb is None:
        notes.append(
            "xgboost python package is not importable in this runtime. Predictions may still work via joblib, but version diagnostics are limited."
        )
        return version_info, notes

    runtime_ver = str(version_info["runtime_xgboost_version"])
    artifact_ver = str(version_info["artifact_xgboost_version"])
    if artifact_ver != "not_stored" and artifact_ver != runtime_ver:
        notes.append(
            f"Artifact was trained with xgboost={artifact_ver}, runtime is xgboost={runtime_ver}. Version mismatch can change predictions."
        )
    elif artifact_ver == "not_stored":
        notes.append(
            "Artifact does not store xgboost training version. Cross-version compatibility cannot be guaranteed."
        )

    # Important for joblib/pickle-loaded boosters across versions.
    model_type = type(model).__name__.lower()
    if "xgb" in model_type:
        notes.append(
            "Model is loaded from serialized Python artifact. For long-term stability, export booster with save_model and load from JSON/UBJ."
        )
    return version_info, notes


def preprocess_dataset(
    df: pd.DataFrame,
    cfg: dict,
    target_col: str,
    dataset_name: str,
    source_rank: int,
) -> pd.DataFrame:
    missing = [c for c in cfg["required_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")

    out = df.copy()
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=["week", "model_family", target_col]).reset_index(drop=True)
    out["_source_rank"] = source_rank
    out = order_rows_for_lag_features(out)
    return out


def previous_weeks(start_week: int, count: int, max_week: int = WEEKS_PER_YEAR) -> list[int]:
    weeks = []
    w = int(start_week)
    for _ in range(count):
        w -= 1
        if w <= 0:
            w = max_week
        weeks.append(w)
    return weeks


def validate_history_requirement(
    historical_df: pd.DataFrame,
    test_df: pd.DataFrame,
    history_weeks_required: int = HISTORY_WEEKS_REQUIRED,
):
    keys = ["model_family"]
    if "model_code" in test_df.columns and "model_code" in historical_df.columns:
        keys.append("model_code")

    problems = []
    hist_groups = historical_df.groupby(keys, dropna=False)

    for grp_key, test_grp in test_df.groupby(keys, dropna=False):
        if grp_key not in hist_groups.groups:
            problems.append(
                {
                    "series": grp_key,
                    "issue": "missing_history_group",
                    "detail": "No matching series found in historical file.",
                }
            )
            continue

        hist_grp = hist_groups.get_group(grp_key).copy()
        test_grp = test_grp.copy()

        if "year" in test_grp.columns and "year" in hist_grp.columns:
            test_grp["year"] = pd.to_numeric(test_grp["year"], errors="coerce")
            hist_grp["year"] = pd.to_numeric(hist_grp["year"], errors="coerce")
            test_grp = test_grp.sort_values(["year", "week"]).reset_index(drop=True)
            hist_grp = hist_grp.sort_values(["year", "week"]).reset_index(drop=True)
            first_test = test_grp.iloc[0]
            hist_before = hist_grp[
                (hist_grp["year"] < first_test["year"])
                | (
                    (hist_grp["year"] == first_test["year"])
                    & (hist_grp["week"] < first_test["week"])
                )
            ]
            if len(hist_before) < history_weeks_required:
                problems.append(
                    {
                        "series": grp_key,
                        "issue": "insufficient_history_rows",
                        "detail": (
                            f"Need {history_weeks_required} rows before first test week "
                            f"({int(first_test['week'])}/{int(first_test['year'])}), got {len(hist_before)}."
                        ),
                    }
                )
        else:
            test_grp = test_grp.sort_values(["_source_rank", "week"]).reset_index(drop=True)
            first_test_week = int(test_grp.iloc[0]["week"])
            required_weeks = previous_weeks(first_test_week, history_weeks_required)
            available_weeks = set(pd.to_numeric(hist_grp["week"], errors="coerce").dropna().astype(int).tolist())
            missing_weeks = [w for w in required_weeks if w not in available_weeks]
            if missing_weeks:
                problems.append(
                    {
                        "series": grp_key,
                        "issue": "missing_required_weeks",
                        "detail": (
                            f"First test week={first_test_week}. Required previous {history_weeks_required} weeks "
                            f"{required_weeks}; missing {missing_weeks}."
                        ),
                    }
                )

    return problems


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    denominator = np.clip(np.abs(y_true), 1, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    accuracy = max(0.0, 100.0 - mape)

    eps = 1e-9
    smape = float(
        np.mean(200.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps))
    )

    mask_10 = np.abs(y_true) >= 10
    if np.any(mask_10):
        mape_gt10 = float(np.mean(np.abs((y_true[mask_10] - y_pred[mask_10]) / y_true[mask_10])) * 100)
        accuracy_gt10 = max(0.0, 100.0 - mape_gt10)
    else:
        mape_gt10 = float("nan")
        accuracy_gt10 = float("nan")

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Accuracy": accuracy,
        "SMAPE": smape,
        "MAPE_gt10": mape_gt10,
        "Accuracy_gt10": accuracy_gt10,
    }


def make_excel_report(out_df: pd.DataFrame, metrics: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame([metrics]).round(4).to_excel(writer, sheet_name="Metrics", index=False)
    return buf.getvalue()


model_label = st.sidebar.selectbox("Model to test", list(MODEL_CONFIG.keys()))
show_debug_audit = st.sidebar.checkbox("Show pipeline debug audit", value=True)
strict_quality_gate = st.sidebar.checkbox(
    "Strict history quality gate",
    value=True,
    help=(
        "If enabled, app will stop when model fit on history rows is too weak. "
        "This helps catch version/unit/runtime mismatches early."
    ),
)
unit_mode = st.sidebar.selectbox(
    "Unit scaling mode",
    ["Auto"] + [str(x) for x in UNIT_FACTOR_CHOICES],
    index=0,
    help=(
        "Use Auto if uploaded files are in large absolute units while model was trained on scaled values. "
        "Predictions/metrics are shown back in original units."
    ),
)
cfg = MODEL_CONFIG[model_label]
requires_lags = cfg.get("requires_lags", True)

try:
    model, trained_features, log_feature_cols, target_col, target_log1p, scaler, artifact_path, artifact = load_artifact(
        model_label
    )
    st.sidebar.success(f"Loaded: {os.path.basename(artifact_path)}")
except Exception as ex:
    st.sidebar.error("Failed to load model artifact")
    st.error(str(ex))
    st.stop()

required_cols_text = ", ".join(f"`{c}`" for c in cfg["required_columns"])
if requires_lags:
    st.info(
        f"Upload TWO files for **{model_label}** testing.\n\n"
        f"1) **Test file** with weeks to evaluate (example: 49 to 59).\n"
        f"2) **Historical file** with at least last ~2 months ({HISTORY_WEEKS_REQUIRED} weeks) actuals before the test start "
        f"so lag/rolling features can be calculated correctly.\n\n"
        f"Required columns in both files: {required_cols_text}\n\n"
        f"Optional: include **`year`** for strict ordering across year boundaries."
    )
else:
    st.info(
        f"Upload the **Test file** for **{model_label}** testing (no history needed - this model uses no lag features).\n\n"
        f"The Historical file box is optional and ignored.\n\n"
        f"Required columns: {required_cols_text}"
    )

uploaded_test = st.file_uploader("Upload TEST dataset (CSV/Excel)", type=["csv", "xlsx", "xls"], key="test_file")
if requires_lags:
    uploaded_history = st.file_uploader(
        "Upload HISTORICAL dataset (CSV/Excel) - prior ~2 months", type=["csv", "xlsx", "xls"], key="hist_file"
    )
else:
    uploaded_history = None

if uploaded_test is None:
    st.stop()
if requires_lags and uploaded_history is None:
    st.stop()

try:
    test_raw = read_uploaded_file(uploaded_test)
    hist_raw = read_uploaded_file(uploaded_history) if uploaded_history is not None else None
except Exception as ex:
    st.error(f"Could not read uploaded file(s): {ex}")
    st.stop()

try:
    test_df = preprocess_dataset(test_raw, cfg, target_col, "Test dataset", source_rank=1)
    if hist_raw is not None:
        historical_df = preprocess_dataset(hist_raw, cfg, target_col, "Historical dataset", source_rank=0)
    else:
        historical_df = test_df.iloc[0:0].copy()
except Exception as ex:
    st.error(str(ex))
    st.stop()

if test_df.empty:
    st.error("No valid rows remain in test dataset after cleaning.")
    st.stop()
if requires_lags and historical_df.empty:
    st.error("No valid rows remain in historical dataset after cleaning.")
    st.stop()

if requires_lags:
    history_issues = validate_history_requirement(historical_df, test_df, HISTORY_WEEKS_REQUIRED)
    if history_issues:
        st.error(
            "Historical file does not satisfy prior-history requirement for lag features. "
            "Please provide at least last ~2 months data before first test week for each series."
        )
        issue_df = pd.DataFrame(
            [
                {
                    "series": str(x["series"]),
                    "issue": x["issue"],
                    "detail": x["detail"],
                }
                for x in history_issues
            ]
        )
        st.dataframe(issue_df, use_container_width=True)
        st.stop()

force_unit_factor = cfg.get("force_unit_factor")
if force_unit_factor is not None:
    # Model declares a fixed unit factor (e.g. CatBoost log1p is scale-native -> factor 1)
    unit_factor = int(force_unit_factor)
    unit_calibration = []
elif unit_mode == "Auto":
    unit_factor, unit_calibration = calibrate_unit_factor(
        historical_df=historical_df,
        target_col=target_col,
        trained_features=trained_features,
        log_feature_cols=log_feature_cols,
        model=model,
        scaler=scaler,
        target_log1p=target_log1p,
    )
    # Fallback heuristic if calibration could not produce usable rows.
    if not any(x["status"] == "ok" and np.isfinite(x["mape"]) for x in unit_calibration):
        unit_factor = suggest_unit_factor(historical_df, target_col)
else:
    unit_factor = int(unit_mode)
    unit_calibration = []

if not historical_df.empty:
    historical_df = apply_unit_scaling(historical_df, unit_factor)
test_df = apply_unit_scaling(test_df, unit_factor)

if requires_lags:
    history_eval_df = add_seasonality(historical_df)
    history_eval_df = add_lag_features(history_eval_df, target_col)
    history_eval_lag_cols = [c for c in history_eval_df.columns if "lag_" in c or "roll_mean" in c]
    history_eval_df = history_eval_df.dropna(subset=history_eval_lag_cols).reset_index(drop=True)
    history_fit = evaluate_model_on_df(
        df=history_eval_df,
        target_col=target_col,
        trained_features=trained_features,
        log_feature_cols=log_feature_cols,
        model=model,
        scaler=scaler,
        target_log1p=target_log1p,
        output_unit_factor=unit_factor,
    )
else:
    history_fit = {"status": "skipped", "rows": 0, "mape": float("nan"), "accuracy": float("nan")}

version_info, xgb_notes = get_xgboost_compatibility_notes(model, artifact)

if requires_lags and strict_quality_gate and history_fit["status"] == "ok" and history_fit["rows"] >= 8 and history_fit["accuracy"] < 50:
    st.error(
        "History quality gate failed: model accuracy on provided historical rows is very low. "
        "This usually means runtime/model-version mismatch, wrong model selection, or input scale mismatch."
    )
    st.write(
        {
            "history_rows_used": history_fit["rows"],
            "history_mape": round(history_fit["mape"], 2),
            "history_accuracy": round(history_fit["accuracy"], 2),
            "unit_factor": unit_factor,
            "runtime_xgboost_version": version_info["runtime_xgboost_version"],
            "artifact_xgboost_version": version_info["artifact_xgboost_version"],
        }
    )
    if xgb_notes:
        for note in xgb_notes:
            st.warning(note)
    st.stop()

if historical_df.empty:
    combined = test_df.copy()
else:
    combined = pd.concat([historical_df, test_df], ignore_index=True)
combined["_is_test"] = combined["_source_rank"].eq(1)

combined = add_seasonality(combined)
combined = add_lag_features(combined, target_col)

lag_cols = [c for c in combined.columns if "lag_" in c or "roll_mean" in c]
before_rows = len(combined)
if requires_lags:
    combined = combined.dropna(subset=lag_cols).reset_index(drop=True)
after_rows = len(combined)

scored = combined[combined["_is_test"]].copy().reset_index(drop=True)
if scored.empty:
    st.error(
        "No test rows remain after feature generation. "
        "Check historical continuity and required columns."
    )
    st.stop()

st.write(f"Rows dropped due to invalid lag features (history + test combined): {before_rows - after_rows}")

missing_features = [f for f in trained_features if f not in scored.columns]
for col in trained_features:
    if col not in scored.columns:
        scored[col] = 0.0

X = scored[trained_features].copy()
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_before_log = X.copy()
X = apply_log_features(X, log_feature_cols)

if scaler is not None:
    X = pd.DataFrame(
        scaler.transform(X),
        columns=trained_features,
        index=X.index,
    )

try:
    pred_model_scale = model.predict(X)
    if target_log1p:
        pred = np.expm1(pred_model_scale)
    else:
        pred = pred_model_scale
    pred = np.clip(pred, 0, None)
    # Optional structural blend (e.g. primary <- inventory conservation: secondary + dbr_stock change)
    cons_col = artifact.get("conservation_col")
    if cons_col and cons_col in scored.columns:
        w_cons = float(artifact.get("conservation_weight", 0.0))
        cons_vals = np.clip(pd.to_numeric(scored[cons_col], errors="coerce").fillna(0).to_numpy(), 0, None)
        pred = (1.0 - w_cons) * pred + w_cons * cons_vals
    pred = np.round(pred * unit_factor).astype(int)
except Exception as ex:
    st.error(f"Prediction failed: {ex}")
    st.stop()

out_cols = [c for c in ["week", "year", "model_code", "model_family"] if c in scored.columns]
out = scored[out_cols].copy()
out["Actual"] = np.round(scored[target_col].values * unit_factor).astype(int)
out["Predicted"] = pred
out["Abs_Error"] = np.abs(out["Actual"] - out["Predicted"])
out["APE(%)"] = (
    (np.abs(out["Actual"] - out["Predicted"]) / np.clip(np.abs(out["Actual"]), 1, None)) * 100
).round(2)

metrics = calculate_metrics(out["Actual"], out["Predicted"])

st.subheader(f"Testing Results - {model_label} model")
st.dataframe(out, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("R2", f"{metrics['R2']:.4f}")
c2.metric("MAPE", f"{metrics['MAPE']:.2f}%")
c3.metric("Accuracy", f"{metrics['Accuracy']:.2f}%")
c4.metric("SMAPE", f"{metrics['SMAPE']:.2f}%")

c5, c6 = st.columns(2)
with c5:
    mape_10 = metrics["MAPE_gt10"]
    txt = "n/a" if np.isnan(mape_10) else f"{mape_10:.2f}%"
    st.metric("MAPE (|Actual|>=10)", txt)
with c6:
    acc_10 = metrics["Accuracy_gt10"]
    txt = "n/a" if np.isnan(acc_10) else f"{acc_10:.2f}%"
    st.metric("Accuracy (|Actual|>=10)", txt)

g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(out["Actual"], out["Predicted"], alpha=0.6, edgecolors="black", s=55)
    mx = float(np.nanmax([out["Actual"].max(), out["Predicted"].max(), 1]))
    ax.plot([0, mx], [0, mx], "r--", lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with g2:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out["Actual"].values, label="Actual")
    ax.plot(out["Predicted"].values, label="Predicted")
    ax.set_xlabel("Row (chronological test rows)")
    ax.set_ylabel("Sales")
    ax.set_title("Trend Fit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

if show_debug_audit:
    with st.expander("Pipeline debug audit (history-aware)", expanded=True):
        st.subheader("DEBUG: Model Expectations")
        st.write("Trained Features:", trained_features)
        st.write("Log Feature Columns:", log_feature_cols)
        st.write("Target column (artifact):", target_col)
        st.write("target_log1p:", target_log1p)
        st.write("RobustScaler in artifact:", scaler is not None)

        st.subheader("DEBUG: Upload Sizes")
        st.write("Historical rows (cleaned):", len(historical_df))
        st.write("Test rows (cleaned):", len(test_df))
        st.write("Scored test rows:", len(scored))
        st.write("Unit factor applied:", unit_factor, f"(mode: {unit_mode})")
        st.write("History fit (same artifact on history rows):", history_fit)
        if unit_calibration:
            st.write("Unit calibration diagnostics (historical rows):")
            st.dataframe(pd.DataFrame(unit_calibration), use_container_width=True)

        st.subheader("DEBUG: Runtime / Artifact Compatibility")
        st.write(version_info)
        if xgb_notes:
            for note in xgb_notes:
                st.warning(note)

        st.subheader("DEBUG: Missing features (before zero-fill)")
        st.write("Missing Features:", missing_features)
        if missing_features:
            st.warning("Some model features were missing and were zero-filled.")

        st.subheader("DEBUG: Lag Feature Sample (test rows)")
        lag_preview_cols = [c for c in [target_col, f"{target_col}_lag_1", f"{target_col}_lag_2", f"{target_col}_lag_4", f"{target_col}_lag_8"] if c in scored.columns]
        show_cols = [c for c in ["week", "year", "model_family"] if c in scored.columns] + lag_preview_cols
        st.dataframe(scored[show_cols].head(15), use_container_width=True)

        st.subheader("DEBUG: Log Transform Check")
        st.write("Columns being log transformed:", log_feature_cols)
        sample_col = log_feature_cols[0] if log_feature_cols else None
        if sample_col and sample_col in X_before_log.columns:
            st.write("Before log:", X_before_log[sample_col].head())
            st.write("After log:", X[sample_col].head())

        st.subheader("DEBUG: Prediction Check")
        st.write("Model raw output (first 10):", np.asarray(pred_model_scale).ravel()[:10])
        st.write("Predictions after inverse transform + clip/round (first 10):", pred[:10])
        st.write("Actuals (first 10):", scored[target_col].head(10).tolist())

for _c in ("_upload_order", "_wk_seg", "_source_rank", "_is_test"):
    if _c in out.columns:
        out = out.drop(columns=[_c])

excel_bytes = make_excel_report(out, metrics)
st.download_button(
    label="Download Excel Report",
    data=excel_bytes,
    file_name=f"{model_label.lower()}_xgboost_lag_feature_test_report_with_history.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)