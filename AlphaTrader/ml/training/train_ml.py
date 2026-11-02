"""
ML Model Trainer — trains XGBoost on historical NSE data.
Feature list is kept in sync with compute_features() in predict.py.

Usage:
    python ml/training/train_ml.py --symbol all --days 1825
    python ml/training/train_ml.py --symbol RELIANCE TCS INFY --days 730
"""

import sys
import argparse
import logging
import warnings
import pickle
from pathlib import Path
from types import ModuleType

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if "numba" not in sys.modules:
    try:
        import numba  # noqa
    except ImportError:
        _s = ModuleType("numba")
        _s.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        sys.modules["numba"] = _s

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("trainer")

MODEL_DIR    = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH   = MODEL_DIR / "xgb_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURES_PATH = MODEL_DIR / "feature_cols.pkl"

# ── Must stay in sync with compute_features() in predict.py ──
# These are the 12 indicators that survived the audit + trim
FEATURE_COLS = [
    # Trend
    "ema_9", "ema_20", "ema_50", "ema_200",
    "supertrend_dir", "adx",
    # Momentum
    "rsi", "macd_hist",
    # Volatility
    "atr_pct", "bb_pct",
    # Volume
    "volume_ratio", "cmf",
    # Structure
    "pct_from_52w_high", "pct_from_52w_low",
    "gap_up", "gap_down",
]

TARGET_UP   =  0.01   # +1% in next 3 days = BUY
TARGET_DOWN = -0.01   # -1% in next 3 days = SELL


def build_target(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    labels = np.where(future_return > TARGET_UP, "BUY",
             np.where(future_return < TARGET_DOWN, "SELL", "HOLD"))
    return pd.Series(labels, index=df.index)


def get_data_with_features(symbol: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV and compute the same features as predict.py."""
    from services.ingestion.adapters.yfinance_adapter import fetch_ohlcv

    df = fetch_ohlcv(symbol, interval="day", days=days)

    # Re-use predict.py's compute_features directly to guarantee feature parity
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from predict import compute_features
    df = compute_features(df)
    df.dropna(inplace=True)
    return df


def train(symbols: list[str], days: int) -> dict:
    all_X, all_y = [], []

    for sym in symbols:
        try:
            log.info("Fetching + computing features for %s ...", sym)
            df = get_data_with_features(sym, days)
            target = build_target(df)
            df["target"] = target
            df.dropna(subset=["target"] + FEATURE_COLS, inplace=True)

            # Only keep features that actually exist in this df
            valid_cols = [c for c in FEATURE_COLS if c in df.columns]
            all_X.append(df[valid_cols].astype(float))
            all_y.append(df["target"])
        except Exception as exc:
            log.warning("Skipping %s: %s", sym, exc)

    if not all_X:
        raise RuntimeError("No data collected — check symbol list and internet connection.")

    X     = pd.concat(all_X).reset_index(drop=True)
    y_raw = pd.concat(all_y).reset_index(drop=True)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # ── Class imbalance fix ───────────────────────────────────
    # HOLD typically dominates (~50-60% of labels).
    # Compute per-class weights so XGBoost treats all classes equally.
    from collections import Counter
    counts  = Counter(y_raw)
    n_total = len(y_raw)
    n_cls   = len(counts)
    # sample_weight: each sample gets weight = n_total / (n_cls * count_of_its_class)
    class_weights = {encoder.transform([cls])[0]: n_total / (n_cls * cnt)
                     for cls, cnt in counts.items()}
    sample_weights = np.array([class_weights[label] for label in y])

    log.info("Dataset: %d samples | class distribution: %s", len(X),
             {k: f"{v/n_total*100:.1f}%" for k, v in counts.items()})

    # ── Walk-forward cross-validation ────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            sample_weight=sample_weights[tr_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = model.predict(X.iloc[val_idx])
        acc   = accuracy_score(y[val_idx], preds)
        cv_scores.append(acc)
        log.info("Fold %d accuracy: %.4f", fold + 1, acc)

    mean_acc = float(np.mean(cv_scores))
    log.info("Mean CV accuracy: %.4f", mean_acc)

    # ── Final model on full data ──────────────────────────────
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, sample_weight=sample_weights)

    # ── Feature importance log ────────────────────────────────
    importance = dict(zip(X.columns, final_model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: -x[1])[:10]
    log.info("Top features: %s", [(f, f"{v:.3f}") for f, v in top])

    # ── Save ──────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(list(X.columns), f)

    log.info("Model saved → %s", MODEL_PATH)
    return {"mean_cv_accuracy": mean_acc, "n_samples": len(X), "features": list(X.columns)}


def main():
    parser = argparse.ArgumentParser(description="AlphaTrader ML Trainer")
    parser.add_argument("--symbol", nargs="+", default=["all"])
    parser.add_argument("--days",   type=int,  default=1825)
    args = parser.parse_args()

    from predict import NIFTY50, NIFTY_MIDCAP, resolve_symbols
    symbols, label = resolve_symbols(args.symbol)
    log.info("Training on %s | %d days", label, args.days)

    result = train(symbols, args.days)
    log.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
