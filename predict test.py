
# UPDATED v4.6 — Adds top toolbar & menu Run/Exit+shortcuts; keeps v4.5 functionality
# ------------------------------------------------------------------------------------
# Changes in v4.6:
#  - Adds toolbar with Run/Exit at the TOP (always visible).
#  - Adds menubar (File → Run / Exit) and shortcuts: Ctrl+R, Ctrl+Q.
#  - Window is resizable and a bit taller by default.
#  - Retains v4.5 features: OOS 5D residual std, date-based subfolder under
#    C:\Users\karanvsi\Desktop\Predictions\watchlist\<YYYY-MM-DD>_<HHMMSS>, liquidity toggle,
#    SHAP fallback, numeric dtype fixes, context columns, predictor-like latest-per-symbol fallback.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import json
import joblib
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Sequence, Set

# ==================== ON-DEMAND PERFORMANCE REPORT ====================
REPORT_WINDOWS = {
    "1D": 1, "5D": 5, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365,
    "2Y": 730, "3Y": 1095, "4Y": 1460, "5Y": 1825,
}

def generate_performance_report(panel, feats, models, window_days):
    cutoff = panel["timestamp"].max() - pd.Timedelta(days=window_days)
    df = panel[panel["timestamp"] >= cutoff].copy()
    if df.empty:
        return None
    X = sanitize_feature_matrix(df[feats].copy())
    p1 = predict_prob_1d(models.get("m1_cls"), X)
    r1p = predict_reg(models.get("m1_reg"), X)
    r3p = predict_reg(models.get("m3_reg"), X)
    r5p = predict_reg(models.get("m5_reg"), X)

    r1 = pd.to_numeric(df["ret_1d_close_pct"], errors="coerce")
    r3 = pd.to_numeric(df["ret_3d_close_pct"], errors="coerce")
    r5 = pd.to_numeric(df["ret_5d_close_pct"], errors="coerce")

    rep = {
        "rows": len(df),
        "hit_1d": float((np.sign(r1p) == np.sign(r1)).mean()),
        "hit_3d": float((np.sign(r3p) == np.sign(r3)).mean()),
        "hit_5d": float((np.sign(r5p) == np.sign(r5)).mean()),
        "avg_pred_1d": float(np.nanmean(r1p)),
        "avg_real_1d": float(np.nanmean(r1)),
        "avg_pred_3d": float(np.nanmean(r3p)),
        "avg_real_3d": float(np.nanmean(r3)),
        "avg_pred_5d": float(np.nanmean(r5p)),
        "avg_real_5d": float(np.nanmean(r5)),
    }
    return rep

def open_performance_report(panel, feats, models):
    win = tk.Toplevel()
    win.title("On-Demand Performance Report")
    ttk.Label(win, text="Select Time Window").pack(pady=5)
    sel = tk.StringVar(value="1M")
    cmb = ttk.Combobox(win, values=list(REPORT_WINDOWS.keys()), textvariable=sel, state="readonly")
    cmb.pack()
    out = tk.Text(win, height=14, width=80)
    out.pack(padx=10, pady=10)

    def run_report():
        w = REPORT_WINDOWS[sel.get()]
        rep = generate_performance_report(panel, feats, models, w)
        out.delete("1.0", tk.END)
        if rep is None:
            out.insert(tk.END, "No data available for selected window.")
            return
        for k, v in rep.items():
            if isinstance(v, float):
                out.insert(tk.END, f"{k:20s}: {v:.4f}\n")
            else:
                out.insert(tk.END, f"{k:20s}: {v}\n")

    ttk.Button(win, text="Generate Report", command=run_report).pack(pady=5)

# ==================== WATCHLIST TRUST SCORE (INLINE) ====================
MAX_STD_5D = 0.035
MAX_MAE_5D = -0.04
WTS_MIN_ACT = 60

def compute_health_score(nan_ratio, schema_mismatch=False, timestamp_issue=False):
    score = 30
    if nan_ratio > 0.05: score -= 10
    if schema_mismatch: score -= 10
    if timestamp_issue: score -= 10
    return max(0, score)

def compute_calibration_score(recent_brier):
    if recent_brier <= 0.18: return 20
    if recent_brier <= 0.22: return 14
    if recent_brier <= 0.26: return 8
    return 0

def compute_drift_score(drift_metric):
    if drift_metric < 0.10: return 15
    if drift_metric < 0.20: return 8
    return 0

def compute_signal_score(prob_1d, pred_ret_1d_pct):
    s = 0
    if prob_1d >= 0.60: s += 10
    if prob_1d >= 0.70: s += 5
    if pred_ret_1d_pct >= 0.015: s += 5
    return s

def compute_risk_score(std_5d, mae_5d):
    s = 15
    if std_5d > MAX_STD_5D: s -= 8
    if mae_5d < MAX_MAE_5D: s -= 7
    return max(0, s)

def compute_wts(row, run_ctx):
    return (
        run_ctx["health"] +
        run_ctx["calibration"] +
        run_ctx["drift"] +
        compute_signal_score(row["prob_up_1d"], row["pred_ret_1d_pct"]) +
        compute_risk_score(row.get("std5", 0), row.get("mae_5d_pct", 0))
    )

# ==================== Pipeline helpers ====================
def sanitize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            s = X[c].astype(str).str.lower()
            uniq = set(s.unique())
            if uniq <= {"true", "false", "nan"}:
                X[c] = s.map({"true": 1, "false": 0}).astype("Int64").fillna(0).astype(int)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        if c.startswith(("CPR_Yday_", "CPR_Tmr_", "Struct_", "DayType_")):
            X[c] = X[c].fillna(0).astype(int)
    return X

def feature_columns_from_panel(panel: pd.DataFrame) -> List[str]:
    cols = list(panel.columns)
    feats = [c for c in cols if (
        c.startswith("D_") or c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_") or
        c.startswith("Struct_") or c.startswith("DayType_")
    )]
    return feats

def load_model(path_str: str):
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        print(f"[WARN] Failed to load {p}: {e}")
        return None

def predict_prob_1d(model, X_df):
    if model is None:
        return np.full(len(X_df), np.nan)
    try:
        proba = model.predict_proba(X_df)[:, 1]
        return proba
    except Exception:
        try:
            raw = model.predict(X_df)
            return 1/(1+np.exp(-np.clip(raw, -10, 10)))
        except Exception:
            return np.full(len(X_df), np.nan)

def predict_reg(model, X_df):
    if model is None:
        return np.full(len(X_df), np.nan)
    try:
        it = getattr(model, "best_iteration_", None)
        return model.predict(X_df, num_iteration=it)
    except Exception:
        try:
            return model.predict(X_df)
        except Exception:
            return np.full(len(X_df), np.nan)

def hit_rate(expected, delivered):
    mask = np.isfinite(expected) & np.isfinite(delivered)
    if mask.sum() == 0: return float('nan')
    return float((np.sign(expected[mask]) == np.sign(delivered[mask])).mean())

def wilson_ci(p_hat, n, z=1.96):
    if n <= 0: return (np.nan, np.nan)
    center = (p_hat + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return (center - half, center + half)

def read_panel_from_dir(panel_dir: Path) -> pd.DataFrame:
    files = sorted(list(panel_dir.glob('*.parquet')))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {panel_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    panel = pd.concat(dfs, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    return panel

def inc_over(a, b):
    """Return incremental (1+a)/(1+b) - 1, robust to NaN."""
    a = np.nan_to_num(a, nan=np.nan)
    b = np.nan_to_num(b, nan=np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (1.0 + a) / (1.0 + b) - 1.0

# ---- Gaussian probability helpers (used for 5D) ----
def _phi_approx(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos_inf = np.isposinf(z); neg_inf = np.isneginf(z)
    out[pos_inf] = 1.0
    out[neg_inf] = 0.0
    finite = np.isfinite(z)
    if np.any(finite):
        x = z[finite]
        t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
        a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
        nd = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * x*x)
        approx = 1.0 - nd * poly
        out[finite] = np.where(x >= 0, approx, 1.0 - approx)
    return out

def prob_up_from_gaussian(mean, std):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    safe_std = np.where(np.isfinite(std) & (std > 0), std, np.nan)
    z = np.divide(mean, safe_std, where=np.isfinite(safe_std))
    prob = _phi_approx(z)
    prob = np.where(np.isfinite(prob), prob, np.where(mean >= 0, 1.0, 0.0))
    return prob

# ---- Symbol/std helpers for 5D prob ----
def _cross_sectional_std(values: np.ndarray, floor: float = 1e-6) -> float:
    s = float(np.nanstd(values))
    return s if np.isfinite(s) and s > floor else 1.0

def _symbol_hist_roll_std(panel: pd.DataFrame, window: int, min_rows: int) -> pd.Series:
    df = panel.sort_values(["symbol","timestamp"])
    grp = df.groupby("symbol", sort=False)
    roll = grp["ret_5d_close_pct"].apply(lambda x: pd.to_numeric(x, errors="coerce").rolling(window, min_periods=min_rows).std())
    df2 = df.copy()
    df2["roll_std_5d"] = roll.values
    last = df2.groupby("symbol", as_index=True)["roll_std_5d"].tail(1)
    return last

def _residual_std_5d(panel: pd.DataFrame, oos5_df: pd.DataFrame) -> Dict[str, float]:
    if oos5_df is None or oos5_df.empty:
        return {}
    idx = oos5_df.get("panel_idx")
    pred = oos5_df.get("oos_pred")
    if idx is None or pred is None:
        return {}
    idx = pd.to_numeric(idx, errors="coerce").astype(int).values
    pred = pd.to_numeric(pred, errors="coerce").values
    in_panel = np.isin(idx, panel.index.values)
    idx = idx[in_panel]; pred = pred[in_panel]
    reals = pd.to_numeric(panel.loc[idx, "ret_5d_close_pct"], errors="coerce").values
    syms = panel.loc[idx, "symbol"].values
    resid = reals - pred
    df = pd.DataFrame({"symbol": syms, "resid": resid})
    stds = df.groupby("symbol")["resid"].std().to_dict()
    return {k: float(v) for k, v in stds.items() if np.isfinite(v) and v > 1e-6}

# ---- SHAP / Importances helpers ----
def compute_shap_or_importance(model, X_df, feature_names):
    """Global explanation: top features over the batch; local top for row 0 if available."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        sv = shap_values if isinstance(shap_values, np.ndarray) else shap_values[0]
        abs_mean = np.nanmean(np.abs(sv), axis=0)
        order = np.argsort(abs_mean)[::-1]
        top_idx = order[:5]
        top_feats = [(feature_names[i], float(abs_mean[i])) for i in top_idx]
        local = None
        if len(X_df) > 0:
            row_sv = sv[0]
            local = sorted([(feature_names[i], float(row_sv[i])) for i in range(len(feature_names))],
                           key=lambda t: abs(t[1]), reverse=True)[:5]
        return {'mode': 'shap', 'top_features': top_feats, 'local_top': local}
    except Exception:
        imp = getattr(model, 'feature_importances_', None)
        if imp is None:
            return {'mode': 'none', 'top_features': [], 'local_top': None}
        arr = np.array(imp)
        order = np.argsort(arr)[::-1]
        top_idx = order[:5]
        top_feats = [(feature_names[i], float(arr[i])) for i in top_idx]
        return {'mode': 'importances', 'top_features': top_feats, 'local_top': None}

def compute_local_shap_strings(model, X_df, symbols: Sequence, topk: int) -> Dict[str, str]:
    """Per-symbol local SHAP strings: 'feat1 +0.12, feat2 -0.08, ...'"""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        sv = shap_values if isinstance(shap_values, np.ndarray) else shap_values[0]
        fnames = list(X_df.columns)
        out = {}
        for i in range(len(X_df)):
            pairs = list(zip(fnames, sv[i]))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            s = ", ".join([f"{n} {v:+.2f}" for n, v in pairs[:topk]])
            out[str(symbols[i])] = s
        return out
    except Exception:
        return {}

# ==================== Ladder stats + empirical calibration ====================
def decile_edges():
    return np.linspace(0.0, 1.0, 11)

def ladder_stats_incremental(df: pd.DataFrame, tgt1=2.0, tgt3=2.0, tgt5=2.0):
    d = df.copy()
    for c in ["prob_up_1d","real_ret_1d_pct","real_ret_3d_pct","real_ret_5d_pct",
              "inc_real_3_over_1","inc_real_5_over_3"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    hit1 = d['real_ret_1d_pct'] >= tgt1
    inc3 = d['inc_real_3_over_1'] >= tgt3
    inc5 = d['inc_real_5_over_3'] >= tgt5

    p1 = float(hit1.mean()) if len(d) else np.nan
    p3g1 = float((inc3[hit1]).mean()) if hit1.any() else np.nan
    p5g3 = float((inc5[inc3 & hit1]).mean()) if (inc3 & hit1).any() else np.nan

    overall = {
        'n_all': int(len(d)),
        'p_1d_hit': p1,
        'n_1d_hit': int(hit1.sum()),
        'p_inc3_given_1d': p3g1,
        'n_for_inc3_given_1d': int((hit1 & d['inc_real_3_over_1'].notna()).sum()),
        'p_inc5_given_3_and_1': p5g3,
        'n_for_inc5_given_3_and_1': int((hit1 & inc3 & d['inc_real_5_over_3'].notna()).sum()),
    }

    edges = decile_edges()
    rows = []
    for i in range(10):
        lo, hi = float(edges[i]), float(edges[i+1])
        b = d[(d['prob_up_1d'] >= lo) & (d['prob_up_1d'] < hi)]
        if len(b) == 0:
            rows.append({'bucket': f'[{lo:.1f},{hi:.1f}]', 'n_all': 0,
                         'p_1d_hit': np.nan, 'p_inc3_given_1d': np.nan,
                         'p_inc5_given_3_and_1': np.nan})
            continue
        h1b = b['real_ret_1d_pct'] >= tgt1
        inc3b = b['inc_real_3_over_1'] >= tgt3
        inc5b = b['inc_real_5_over_3'] >= tgt5
        rows.append({
            'bucket': f'[{lo:.1f},{hi:.1f}]',
            'n_all': int(len(b)),
            'p_1d_hit': float(h1b.mean()),
            'p_inc3_given_1d': float((inc3b[h1b]).mean()) if h1b.any() else np.nan,
            'p_inc5_given_3_and_1': float((inc5b[inc3b & h1b]).mean()) if (inc3b & h1b).any() else np.nan,
        })
    return overall, pd.DataFrame(rows)

def build_prob_calibration(df: pd.DataFrame, tgt1=2.0):
    d = df.copy()
    for c in ["prob_up_1d","real_ret_1d_pct"]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d['real_ret_1d_pct'].notna() & d['prob_up_1d'].notna()]
    edges = decile_edges()
    out = []
    for i in range(10):
        lo, hi = float(edges[i]), float(edges[i+1])
        b = d[(d['prob_up_1d'] >= lo) & (d['prob_up_1d'] < hi)]
        n = len(b)
        if n == 0:
            out.append({'lo': lo, 'hi': hi, 'p': np.nan, 'n': 0, 'ci_low': np.nan, 'ci_high': np.nan})
            continue
        p = float((b['real_ret_1d_pct'] >= tgt1).mean())
        ci_low, ci_high = wilson_ci(p, n)
        out.append({'lo': lo, 'hi': hi, 'p': p, 'n': n, 'ci_low': ci_low, 'ci_high': ci_high})
    calib = pd.DataFrame(out)
    return calib

def map_prob_to_empirical(prob: float, calib_df: pd.DataFrame) -> float:
    if not np.isfinite(prob) or calib_df is None or calib_df.empty:
        return np.nan
    for _, r in calib_df.iterrows():
        if prob >= r['lo'] and prob < r['hi']:
            return float(r['p']) if np.isfinite(r['p']) else np.nan
    last = calib_df.iloc[-1]
    if prob >= last['lo']:
        return float(last['p']) if np.isfinite(last['p']) else np.nan
    return np.nan

# ==================== NSE holiday calendar (2025) ====================
NSE_HOLIDAYS_2025: Set[pd.Timestamp] = set(pd.to_datetime(d) for d in [
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
])

def next_trading_date(start_date: pd.Timestamp, holidays: Set[pd.Timestamp]) -> pd.Timestamp:
    d = start_date + pd.Timedelta(days=1)
    while d.weekday() >= 5 or d.normalize() in holidays:  # 5=Sat, 6=Sun
        d += pd.Timedelta(days=1)
    return d.normalize()

# ==================== NEW WATCHLIST (enhanced) ====================
DEFAULT_M1CLS      = r"C:\Users\karanvsi\PyCharmMiscProject\model_1d_cls_calib.joblib"
DEFAULT_M3REG      = r"C:\Users\karanvsi\PyCharmMiscProject\model_3d_rf.joblib"
DEFAULT_M5REG      = r"C:\Users\karanvsi\PyCharmMiscProject\model_5d_rf.joblib"
DEFAULT_PANEL_DIR  = r"C:\Users\karanvsi\PyCharmMiscProject\panel_by_year"
DEFAULT_OOS5_PATH  = r"C:\Users\karanvsi\PyCharmMiscProject\oos_preds_5d.csv"

DEFAULT_NEXTDAY_DIR   = Path.home() / "Desktop" / "Kite Connect" / "NextDay Watchlist"
WATCHLIST_BASE_DIR    = Path(r"C:\Users\karanvsi\Desktop\Predictions\watchlist")
DEFAULT_TOPK          = 3  # SHAP top-k

# Picks thresholds
PICKS_CONF_MIN   = 0.60
PICKS_PRED1D_MIN = 0.015  # +1.5%

# Default liquidity (aligned with predictor.py)
MIN_CLOSE_DEFAULT     = 2.0
MIN_AVG20_VOL_DEFAULT = 200_000

# Probability std controls
PROB_STD_METHOD   = "residual"
PROB_STD_WINDOW   = 252
PROB_STD_MIN_ROWS = 60

def build_new_watchlist(
    panel: pd.DataFrame,
    feats: list,
    models: dict,
    upcoming_trading_date: pd.Timestamp,
    calib_df: Optional[pd.DataFrame],
    conv_df: Optional[pd.DataFrame],
    save_dir: Path,
    panel_last_date: Optional[pd.Timestamp] = None,
    tgt1=2.0, tgt3=2.0, tgt5=2.0,
    model_conv3=None, model_conv5=None,
    calibrator_pickle: Optional[str] = None,
    use_local_shap: bool = True,
    shap_topk: int = DEFAULT_TOPK,
    prob_std_method: str = PROB_STD_METHOD,
    prob_std_window: int = PROB_STD_WINDOW,
    prob_std_min_rows: int = PROB_STD_MIN_ROWS,
    oos5_path: Optional[str] = None,
    apply_liquidity: bool = True,
    min_close: float = MIN_CLOSE_DEFAULT,
    min_avg20_vol: int = MIN_AVG20_VOL_DEFAULT,
    watch_dir: Optional[Path] = None
):
    """
    Build detailed watchlist + compact next-day export + picks.
    Adds LastDataAvailableDate and PredictionForDate columns.
    Saves to default NextDay folder AND to watch_dir (date-based subfolder).
    Returns paths from watch_dir (primary outputs).
    """
    qdate = upcoming_trading_date.date()

    panel_sorted = panel.sort_values(["symbol", "timestamp"]).copy()
    panel_sorted["avg20_vol"] = panel_sorted.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )

    mask = panel_sorted["timestamp"].dt.date == qdate
    day_df = panel_sorted.loc[mask].copy()
    if day_df.empty:
        day_df = panel_sorted.groupby("symbol", as_index=False).tail(1).copy()

    X = sanitize_feature_matrix(day_df[feats].copy())
    prob_up_1d = predict_prob_1d(models.get('m1_cls'), X)
    pred_ret_1d = predict_reg(models.get('m1_reg'), X)
    pred_ret_3d = predict_reg(models.get('m3_reg'), X)
    pred_ret_5d = predict_reg(models.get('m5_reg'), X)

    std5 = np.full(len(day_df), np.nan, dtype=float)
    if prob_std_method == "residual" and oos5_path:
        try:
            pth = Path(oos5_path)
            if pth.exists():
                oos5_df = pd.read_csv(pth)
                residual_map = _residual_std_5d(panel_sorted, oos5_df)
                std5 = day_df["symbol"].map(residual_map).astype(float).values
            else:
                print(f"[WARN] OOS 5D file not found at: {oos5_path}")
        except Exception as e:
            print(f"[WARN] Residual std failed: {e}")

    if np.isnan(std5).mean() > 0.5 and prob_std_method in {"residual", "symbol_hist"}:
        sym_hist = _symbol_hist_roll_std(panel_sorted, window=int(prob_std_window), min_rows=int(prob_std_min_rows))
        std_map2 = sym_hist.to_dict()
        hist_std = day_df["symbol"].map(std_map2).astype(float).values
        std5 = np.where(np.isfinite(std5), std5, hist_std)
    if np.isnan(std5).mean() > 0.5:
        cs = _cross_sectional_std(pred_ret_5d)
        std5 = np.where(np.isfinite(std5), std5, cs)

    prob_up_5d = prob_up_from_gaussian(pred_ret_5d, std5)

    if calib_df is not None and len(prob_up_1d) > 0:
        p1_emp = np.array([map_prob_to_empirical(p, calib_df) for p in prob_up_1d])
        p1_emp = np.where(np.isfinite(p1_emp), p1_emp, prob_up_1d)
    else:
        p1_emp = prob_up_1d

    def conv_inc3_from_prob(prob):
        if conv_df is None or conv_df.empty:
            return np.nan
        def parse_bucket(b: str) -> tuple:
            s = str(b).strip()
            if s and s[0] in '([' and s[-1] in ')]':
                s = s[1:-1]
                parts = s.split(',')
                if len(parts) != 2:
                    return (np.nan, np.nan)
                try:
                    lo = float(parts[0].strip()); hi = float(parts[1].strip())
                    return (lo, hi)
                except Exception:
                    return (np.nan, np.nan)
            return (np.nan, np.nan)
        for _, r in conv_df.iterrows():
            lo, hi = parse_bucket(r['bucket'])
            if np.isfinite(lo) and np.isfinite(hi) and (prob >= lo) and (prob < hi):
                return float(r.get('p_inc3_given_1d', np.nan))
        last = conv_df.iloc[-1]
        return float(last.get('p_inc3_given_1d', np.nan))
    p3_conv = np.array([conv_inc3_from_prob(p) for p in prob_up_1d])

    k = 0.7
    mom_context = np.zeros(len(day_df))
    quality = np.ones(len(day_df))
    overext = np.zeros(len(day_df))
    score_d1 = (0.50 * p1_emp
                + 0.25 * (1/(1+np.exp(-k * np.nan_to_num(pred_ret_1d))))
                + 0.15 * mom_context
                + 0.10 * quality
                - 0.10 * overext)

    p_next2 = 1 - (1 - np.clip(p1_emp, 0, 1)) * (1 - np.clip(p3_conv, 0, 1))

    feature_names = list(X.columns)
    expl = compute_shap_or_importance(models.get('m3_reg') or models.get('m1_cls'), X, feature_names)

    shap_map = {}
    if use_local_shap and models.get('m1_cls') is not None:
        try:
            shap_map = compute_local_shap_strings(models.get('m1_cls'), X, day_df["symbol"].values, shap_topk)
        except Exception as e:
            print(f"[WARN] Local SHAP failed: {e}")
            global_str = ", ".join([f"{n}:{v:.3f}" for n, v in expl.get('top_features', [])])
            shap_map = {str(s): global_str for s in day_df["symbol"].values}

    out = day_df[["symbol","timestamp","close","volume","avg20_vol"]].copy()
    out["prob_up_1d"] = prob_up_1d
    out["pred_ret_1d_pct"] = pred_ret_1d
    out["pred_ret_3d_pct"] = pred_ret_3d
    out["pred_ret_5d_pct"] = pred_ret_5d
    out["pred_std_5d"] = std5
    out["prob_up_5d"] = prob_up_5d
    out["inc_pred_3_over_1"] = inc_over(pred_ret_3d, pred_ret_1d)
    out["inc_pred_5_over_3"] = inc_over(pred_ret_5d, pred_ret_3d)
    out["P1_empirical"] = p1_emp
    out["P_inc3_given_1_empirical"] = p3_conv
    out["ImmediatePickScore_D1"] = score_d1
    out["P_hit_next_2d"] = p_next2
    out["ExplainerMode"] = expl.get('mode', 'none')
    out["SHAP_or_Importances"] = ", ".join([f"{n}:{v:.3f}" for n, v in expl.get('top_features', [])])

    pre_rows = len(out)
    if apply_liquidity:
        liquid_mask = (out["close"] >= float(min_close)) & (out["avg20_vol"] >= float(min_avg20_vol))
        if liquid_mask.any():
            out = out.loc[liquid_mask].copy()
    post_rows = len(out)
    print(f"[INFO] Liquidity filter: {pre_rows} -> {post_rows} rows "
          f"(min_close={min_close}, min_avg20_vol={min_avg20_vol}, applied={apply_liquidity})")

    DEFAULT_NEXTDAY_DIR.mkdir(parents=True, exist_ok=True)
    if watch_dir is not None:
        watch_dir.mkdir(parents=True, exist_ok=True)

    generated_at = pd.Timestamp(dt.datetime.now()).tz_localize(None)

    for c in ["pred_ret_1d_pct","pred_ret_3d_pct","pred_ret_5d_pct","pred_std_5d","prob_up_5d","prob_up_1d"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    shap_series = out["symbol"].map(lambda s: shap_map.get(str(s), ""))

    last_date = pd.to_datetime(panel_last_date).date() if panel_last_date is not None else None
    prediction_for = upcoming_trading_date.normalize().date()

    out_compact = pd.DataFrame({
        "symbol": out["symbol"],
        "WC NextTradingDate": upcoming_trading_date.normalize(),
        "LastDataAvailableDate": last_date,
        "PredictionForDate": prediction_for,
        "GeneratedAt": generated_at,
        "close": out["close"].astype(float),
        "volume": out["volume"].astype(float),
        "avg20_vol": out["avg20_vol"].astype(float),
        "pred_ret_1d_pct": out["pred_ret_1d_pct"].astype(float),
        "pred_ret_3d_pct": out["pred_ret_3d_pct"].astype(float),
        "pred_ret_5d_pct": out["pred_ret_5d_pct"].astype(float),
        "pred_std_5d": out["pred_std_5d"].astype(float),
        "prob_up_5d": out["prob_up_5d"].astype(float),
        "Confidence_1d_raw": out["prob_up_1d"].astype(float),
        "SHAP_top": shap_series,
    })

    calibrated = None
    if calibrator_pickle:
        try:
            calibrator = joblib.load(calibrator_pickle)
            probs = out_compact["Confidence_1d_raw"].to_numpy()
            arr = probs.reshape(-1, 1) if probs.ndim == 1 else probs
            out_compact["Confidence_1d_calib"] = calibrator.predict_proba(arr)[:, 1]
            calibrated = out_compact["Confidence_1d_calib"].to_numpy()
        except Exception as e:
            print(f"[WARN] Calibration failed: {e}")

    compact_csv_default = DEFAULT_NEXTDAY_DIR / "watchlist_nextday.csv"
    compact_xlsx_default = DEFAULT_NEXTDAY_DIR / "watchlist_nextday.xlsx"
    out_compact.to_csv(compact_csv_default, index=False)
    try:
        out_compact.to_excel(compact_xlsx_default, index=False, engine="openpyxl")
    except Exception as e:
        print(f"[WARN] Excel write failed ({e}); CSV saved.")

    conf_for_filter = calibrated if calibrated is not None else out_compact["Confidence_1d_raw"].to_numpy()
    mask_picks = (conf_for_filter >= PICKS_CONF_MIN) & (out_compact["pred_ret_1d_pct"].to_numpy() >= PICKS_PRED1D_MIN)
    picks_df = out_compact.loc[mask_picks].copy().sort_values(
        by=["Confidence_1d_calib" if "Confidence_1d_calib" in out_compact.columns else "Confidence_1d_raw",
            "pred_ret_1d_pct"],
        ascending=[False, False]
    )
    picks_default = DEFAULT_NEXTDAY_DIR / "watchlist_nextday_picks.csv"
    picks_df.to_csv(picks_default, index=False)

    ts_suffix = dt.datetime.now().strftime("%H%M%S")
    detailed_default = DEFAULT_NEXTDAY_DIR / f"watchlist_ladder_{qdate}_{ts_suffix}.csv"
    out.sort_values(["ImmediatePickScore_D1", "P_hit_next_2d"], ascending=[False, False]).to_csv(detailed_default, index=False)

    if watch_dir is not None:
        compact_csv_watch = watch_dir / "watchlist_nextday.csv"
        compact_xlsx_watch = watch_dir / "watchlist_nextday.xlsx"
        picks_watch = watch_dir / "watchlist_nextday_picks.csv"
        detailed_watch = watch_dir / f"watchlist_ladder_{qdate}_{ts_suffix}.csv"

        out_compact.to_csv(compact_csv_watch, index=False)
        try:
            out_compact.to_excel(compact_xlsx_watch, index=False, engine="openpyxl")
        except Exception as e:
            print(f"[WARN] Excel write failed ({e}); CSV saved in watch_dir.")
        picks_df.to_csv(picks_watch, index=False)
        out.sort_values(["ImmediatePickScore_D1","P_hit_next_2d"], ascending=[False, False]).to_csv(detailed_watch, index=False)

        return str(detailed_watch), str(compact_csv_watch), str(compact_xlsx_watch), str(picks_watch)

    return str(detailed_default), str(compact_csv_default), str(compact_xlsx_default), str(picks_default)

# ==================== GUI ====================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Daily Metrics (GUI v4.6, NSE-aware)")
        self.geometry("1180x980")     # taller & wider
        self.resizable(True, True)    # allow resizing

        # --- Menubar with Run/Exit and shortcuts ---
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Run", accelerator="Ctrl+R", command=self.run)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)
        self.bind_all("<Control-r>", lambda e: self.run())
        self.bind_all("<Control-q>", lambda e: self.destroy())

        # --- Top toolbar (always visible) ---
        header = ttk.Frame(self, padding=8)
        header.pack(fill="x")
        ttk.Button(header, text="Run", command=self.run).pack(side="left", padx=(0,8))
        ttk.Button(header, text="Exit", command=self.destroy).pack(side="left")

        # Main content frame
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        # Inputs
        self.panel_dir_var = tk.StringVar(value=DEFAULT_PANEL_DIR)
        self.m1cls_var = tk.StringVar(value=DEFAULT_M1CLS)
        self.m1reg_var = tk.StringVar(value="")   # optional 1D regressor
        self.m3reg_var = tk.StringVar(value=DEFAULT_M3REG)
        self.m5reg_var = tk.StringVar(value=DEFAULT_M5REG)
        self.oos5_path_var = tk.StringVar(value=DEFAULT_OOS5_PATH)

        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        self.upcoming_date_var = tk.StringVar()   # if blank, auto-compute NextTradingDate

        self.min_prob_var = tk.StringVar(value="0.60")
        self.tgt1_var = tk.StringVar(value="2.0")
        self.tgt3_var = tk.StringVar(value="2.0")
        self.tgt5_var = tk.StringVar(value="2.0")
        self.alpha_tol_var = tk.StringVar(value="0.8")
        self.mode_var = tk.StringVar(value="new_watchlist")  # default to new_watchlist

        # Calibrator & SHAP options
        self.calibrator_var = tk.StringVar(value="")
        self.use_holiday_calendar_var = tk.BooleanVar(value=True)  # NSE-aware next trading date
        self.use_local_shap_var = tk.BooleanVar(value=True)
        self.shap_topk_var = tk.StringVar(value=str(DEFAULT_TOPK))

        # Liquidity filter controls
        self.apply_liquidity_var = tk.BooleanVar(value=True)
        self.min_close_var = tk.StringVar(value=str(MIN_CLOSE_DEFAULT))
        self.min_avg20_var = tk.StringVar(value=str(MIN_AVG20_VOL_DEFAULT))

        # Mode selector
        ttk.Label(frm, text="Run mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frm, text="Back-test", variable=self.mode_var, value="backtest").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frm, text="New Watchlist", variable=self.mode_var, value="new_watchlist").grid(row=0, column=2, sticky="w")

        # Panel dir
        ttk.Label(frm, text="Panel directory (parquet by year):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.panel_dir_var, width=64).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self.choose_panel_dir).grid(row=1, column=2, padx=6)

        # Models
        ttk.Label(frm, text="1D Classifier joblib:").grid(row=2, column=0, sticky="w", pady=(8,0))
        ttk.Entry(frm, textvariable=self.m1cls_var, width=64).grid(row=2, column=1, sticky="we", pady=(8,0))
        ttk.Button(frm, text="Browse", command=lambda: self.pick_model(self.m1cls_var)).grid(row=2, column=2, padx=6, pady=(8,0))

        ttk.Label(frm, text="1D Regressor joblib (optional):").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.m1reg_var, width=64).grid(row=3, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=lambda: self.pick_model(self.m1reg_var)).grid(row=3, column=2, padx=6)

        ttk.Label(frm, text="3D Regressor joblib:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.m3reg_var, width=64).grid(row=4, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=lambda: self.pick_model(self.m3reg_var)).grid(row=4, column=2, padx=6)

        ttk.Label(frm, text="5D Regressor joblib:").grid(row=5, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.m5reg_var, width=64).grid(row=5, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=lambda: self.pick_model(self.m5reg_var)).grid(row=5, column=2, padx=6)

        # OOS 5D path (for residual std)
        ttk.Label(frm, text="OOS 5D CSV (oos_preds_5d.csv):").grid(row=6, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.oos5_path_var, width=64).grid(row=6, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self.pick_oos5).grid(row=6, column=2, padx=6)

        # Dates & thresholds
        ttk.Label(frm, text="Start date (YYYY-MM-DD):").grid(row=7, column=0, sticky="w", pady=(8,0))
        ttk.Entry(frm, textvariable=self.start_date_var, width=20).grid(row=7, column=1, sticky="w", pady=(8,0))

        ttk.Label(frm, text="End date (YYYY-MM-DD):").grid(row=8, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.end_date_var, width=20).grid(row=8, column=1, sticky="w")

        ttk.Label(frm, text="Upcoming date (YYYY-MM-DD) [optional]:").grid(row=9, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.upcoming_date_var, width=20).grid(row=9, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Use NSE holiday calendar (compute NextTradingDate)", variable=self.use_holiday_calendar_var).grid(row=9, column=2, sticky="w")

        ttk.Label(frm, text="Min prob 1D (0.00–1.00):").grid(row=10, column=0, sticky="w", pady=(8,0))
        ttk.Entry(frm, textvariable=self.min_prob_var, width=10).grid(row=10, column=1, sticky="w", pady=(8,0))

        ttk.Label(frm, text="Targets (%): T1 (1D), T3 (3↦1 inc), T5 (5↦3 inc)").grid(row=11, column=0, sticky="w")
        tgrid = ttk.Frame(frm); tgrid.grid(row=11, column=1, sticky="w")
        ttk.Entry(tgrid, textvariable=self.tgt1_var, width=6).grid(row=0, column=0)
        ttk.Entry(tgrid, textvariable=self.tgt3_var, width=6).grid(row=0, column=1, padx=6)
        ttk.Entry(tgrid, textvariable=self.tgt5_var, width=6).grid(row=0, column=2)

        ttk.Label(frm, text="Pred tolerance α (met pred when R ≥ α·P)").grid(row=12, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.alpha_tol_var, width=6).grid(row=12, column=1, sticky="w")

        # Calibrator & SHAP
        ttk.Label(frm, text="Calibrator pickle (optional):").grid(row=13, column=0, sticky="w", pady=(6,0))
        ttk.Entry(frm, textvariable=self.calibrator_var, width=64).grid(row=13, column=1, sticky="we", pady=(6,0))
        ttk.Button(frm, text="Browse", command=lambda: self.pick_model(self.calibrator_var)).grid(row=13, column=2, padx=6, pady=(6,0))

        ttk.Checkbutton(frm, text="Compute local SHAP per symbol", variable=self.use_local_shap_var).grid(row=14, column=0, sticky="w")
        ttk.Label(frm, text="Top‑K SHAP (default 3):").grid(row=14, column=1, sticky="w")
        ttk.Entry(frm, textvariable=self.shap_topk_var, width=6).grid(row=14, column=2, sticky="w")

        # Liquidity filter block
        sep = ttk.Separator(frm); sep.grid(row=15, column=0, columnspan=3, sticky="we", pady=(10,6))
        ttk.Checkbutton(frm, text="Apply liquidity filter", variable=self.apply_liquidity_var).grid(row=16, column=0, sticky="w")
        liq_grid = ttk.Frame(frm); liq_grid.grid(row=16, column=1, sticky="w")
        ttk.Label(liq_grid, text="Min Close:").grid(row=0, column=0, padx=(0,4))
        ttk.Entry(liq_grid, textvariable=self.min_close_var, width=8).grid(row=0, column=1, padx=(0,12))
        ttk.Label(liq_grid, text="Min Avg20Vol:").grid(row=0, column=2, padx=(0,4))
        ttk.Entry(liq_grid, textvariable=self.min_avg20_var, width=10).grid(row=0, column=3)

        # Performance report button
        self.btn_perf = ttk.Button(frm, text="Open Performance Report", command=self.open_perf_later, state="disabled")
        self.btn_perf.grid(row=17, column=0, sticky="w", pady=(8,0))

        # Progress & status
        self.pb = ttk.Progressbar(frm, orient="horizontal", length=980, mode="determinate")
        self.pb.grid(row=18, column=0, columnspan=3, pady=(16,6))
        self.status = tk.Text(frm, height=18, width=132)
        self.status.grid(row=19, column=0, columnspan=3, sticky="we")
        self.status.configure(state="disabled")

        # Bottom buttons (redundant; toolbar already has Run/Exit)
        btnfrm = ttk.Frame(frm); btnfrm.grid(row=20, column=0, columnspan=3, pady=(12,0))
        ttk.Button(btnfrm, text="Run", command=self.run).grid(row=0, column=0, padx=6)
        ttk.Button(btnfrm, text="Exit", command=self.destroy).grid(row=0, column=1, padx=6)

        frm.columnconfigure(1, weight=1)

        # Placeholders for loaded objects to enable performance report
        self._panel = None
        self._feats = None
        self._models = None

    def choose_panel_dir(self):
        d = filedialog.askdirectory(title="Select panel directory")
        if d:
            self.panel_dir_var.set(d)

    def pick_model(self, var):
        f = filedialog.askopenfilename(title="Select model file",
            filetypes=[("Model files","*.joblib;*.pkl"), ("All","*.*")])
        if f:
            var.set(f)

    def pick_oos5(self):
        f = filedialog.askopenfilename(title="Select oos_preds_5d.csv",
            filetypes=[("CSV","*.csv"), ("All","*.*")])
        if f:
            self.oos5_path_var.set(f)

    def log(self, msg: str):
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")
        self.update_idletasks()

    def open_perf_later(self):
        if self._panel is not None and self._feats is not None and self._models is not None:
            open_performance_report(self._panel, self._feats, self._models)
        else:
            messagebox.showwarning("Not ready", "Load a panel and models via Run first.")

    def run(self):
        try:
            panel_dir = Path(self.panel_dir_var.get().strip())
            if not panel_dir.exists():
                messagebox.showerror("Error", "Panel directory does not exist.")
                return

            base_preds_dir = Path(r"C:\Users\karanvsi\Desktop\Predictions")
            base_preds_dir.mkdir(parents=True, exist_ok=True)

            # Dates
            try:
                start = pd.to_datetime(self.start_date_var.get().strip()).date() if self.start_date_var.get().strip() else None
                end   = pd.to_datetime(self.end_date_var.get().strip()).date() if self.end_date_var.get().strip() else None
            except Exception:
                messagebox.showerror("Error", "Invalid start/end date. Use YYYY-MM-DD.")
                return

            # Load panel & features
            self.log(f"Loading panel from: {panel_dir}")
            panel = read_panel_from_dir(panel_dir)
            feats = feature_columns_from_panel(panel)
            self.log(f"Feature columns detected: {len(feats)}")

            # Last data date
            panel_last_ts = pd.to_datetime(panel["timestamp"]).max()
            panel_last_date = panel_last_ts.normalize()
            self.log(f"Last data available date in panel: {panel_last_date.date()}")

            # Load models
            models = {
                'm1_cls': load_model(self.m1cls_var.get().strip()),
                'm1_reg': load_model(self.m1reg_var.get().strip()),
                'm3_reg': load_model(self.m3reg_var.get().strip()),
                'm5_reg': load_model(self.m5reg_var.get().strip()),
            }
            for k, v in models.items():
                self.log(f"{k}: {'OK' if v is not None else 'MISSING'}")

            # Enable performance report
            self._panel = panel; self._feats = feats; self._models = models
            self.btn_perf.configure(state="normal")

            # Parse numerics
            try:
                min_prob = float(self.min_prob_var.get().strip())
                tgt1 = float(self.tgt1_var.get().strip())
                tgt3 = float(self.tgt3_var.get().strip())
                tgt5 = float(self.tgt5_var.get().strip())
                alpha_tol = float(self.alpha_tol_var.get().strip())
                shap_topk = int(self.shap_topk_var.get().strip() or DEFAULT_TOPK)
            except Exception:
                messagebox.showerror("Error", "Invalid numeric input(s).")
                return

            # Optional historical calibration
            calib_df = None
            conv_df = None
            model_conv3 = None
            model_conv5 = None

            if start and end:
                if start > end:
                    messagebox.showerror("Error", "Start date must be ≤ End date.")
                    return
                cur = start
                all_rows = []
                while cur <= end:
                    mask = panel["timestamp"].dt.date == cur
                    day_df = panel.loc[mask]
                    if len(day_df) == 0:
                        cur += dt.timedelta(days=1)
                        continue
                    Xd = sanitize_feature_matrix(day_df[feats].copy())
                    prob_up_1d = predict_prob_1d(models.get('m1_cls'), Xd)
                    tmp = day_df[["symbol","timestamp"]].copy()
                    tmp["prob_up_1d"] = prob_up_1d
                    tmp["real_ret_1d_pct"] = pd.to_numeric(day_df.get("ret_1d_close_pct"), errors='coerce').values
                    tmp["real_ret_3d_pct"] = pd.to_numeric(day_df.get("ret_3d_close_pct"), errors='coerce').values
                    tmp["real_ret_5d_pct"] = pd.to_numeric(day_df.get("ret_5d_close_pct"), errors='coerce').values
                    tmp["inc_real_3_over_1"] = inc_over(tmp["real_ret_3d_pct"], tmp["real_ret_1d_pct"])
                    tmp["inc_real_5_over_3"] = inc_over(tmp["real_ret_5d_pct"], tmp["real_ret_3d_pct"])
                    all_rows.append(tmp)
                    cur += dt.timedelta(days=1)

                if len(all_rows) > 0:
                    hist_df = pd.concat(all_rows, ignore_index=True)
                    calib_df = build_prob_calibration(hist_df, tgt1=tgt1)
                    overall_inc, conv_df = ladder_stats_incremental(hist_df, tgt1=tgt1, tgt3=tgt3, tgt5=tgt5)
                    ts_suffix = dt.datetime.now().strftime("%H%M%S")
                    calib_path = base_preds_dir / f"prob_calibration_{start}_{end}_{ts_suffix}.csv"
                    conv_path  = base_preds_dir / f"ladder_incremental_by_bucket_{start}_{end}_{ts_suffix}.csv"
                    calib_df.to_csv(calib_path, index=False)
                    conv_df.to_csv(conv_path, index=False)
                    self.log(f"Saved calibration & conversion stats:\n  {calib_path}\n  {conv_path}")

                    model_conv3, model_conv5 = train_follow_through_models(panel, feats,
                                            start=start, end=end,
                                            tgt1=tgt1, tgt3=tgt3, tgt5=tgt5, alpha_tol=alpha_tol)
                    self.log(f"Follow-through models: FT3={'OK' if model_conv3 else 'MISSING'} "
                             f"FT5={'OK' if model_conv5 else 'MISSING'}")

            # Resolve upcoming trading date
            upcoming_input = self.upcoming_date_var.get().strip()
            if upcoming_input:
                base = pd.to_datetime(upcoming_input).normalize()
                upcoming_trading = next_trading_date(base, NSE_HOLIDAYS_2025) if self.use_holiday_calendar_var.get() else base
            else:
                base = pd.to_datetime(panel_last_date).normalize()
                upcoming_trading = next_trading_date(base, NSE_HOLIDAYS_2025) if self.use_holiday_calendar_var.get() else (base + pd.Timedelta(days=1))
            self.upcoming_date_var.set(upcoming_trading.date().isoformat())
            self.log(f"PredictionForDate (NSE): {upcoming_trading.date()} | LastDataAvailableDate: {panel_last_date.date()}")

            # Create unique watchlist subfolder
            subdir = WATCHLIST_BASE_DIR / f"{upcoming_trading.date().isoformat()}_{dt.datetime.now().strftime('%H%M%S')}"
            subdir.mkdir(parents=True, exist_ok=True)
            self.log(f"Watchlist folder: {subdir}")

            # Liquidity settings
            try:
                apply_liquidity = bool(self.apply_liquidity_var.get())
                min_close = float(self.min_close_var.get().strip())
                min_avg20 = float(self.min_avg20_var.get().strip())
            except Exception:
                messagebox.showerror("Error", "Invalid liquidity inputs.")
                return

            oos5_path = self.oos5_path_var.get().strip() or None

            # Build watchlist (+ picks), write to subdir & mirror to NextDay folder
            detailed_path, compact_csv, compact_xlsx, picks_path = build_new_watchlist(
                panel, feats, models,
                upcoming_trading_date=upcoming_trading,
                calib_df=calib_df, conv_df=conv_df, save_dir=base_preds_dir,
                panel_last_date=panel_last_date,
                tgt1=tgt1, tgt3=tgt3, tgt5=tgt5,
                model_conv3=model_conv3, model_conv5=model_conv5,
                calibrator_pickle=(self.calibrator_var.get().strip() or None),
                use_local_shap=self.use_local_shap_var.get(),
                shap_topk=int(self.shap_topk_var.get().strip() or DEFAULT_TOPK),
                prob_std_method=PROB_STD_METHOD,
                prob_std_window=PROB_STD_WINDOW,
                prob_std_min_rows=PROB_STD_MIN_ROWS,
                oos5_path=oos5_path,
                apply_liquidity=apply_liquidity,
                min_close=min_close,
                min_avg20_vol=min_avg20,
                watch_dir=subdir
            )

            self.log(f"Saved new watchlist (detailed): {detailed_path}")
            self.log(f"Saved compact next-day files:\n  {compact_csv}\n  {compact_xlsx}")
            self.log(f"Picks file: {picks_path}")
            self.log(f"(Also mirrored to Desktop\\Kite Connect\\NextDay Watchlist)")

            messagebox.showinfo("Done",
                f"Watchlist folder:\n  {subdir}\n\n"
                f"Also mirrored to:\n  {DEFAULT_NEXTDAY_DIR}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    App().mainloop()
