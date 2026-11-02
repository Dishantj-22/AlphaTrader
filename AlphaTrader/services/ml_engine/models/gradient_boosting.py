"""
XGBoost / LightGBM gradient boosting model for next-bar return prediction.
Predicts probability of: UP (>0.5%), DOWN (<-0.5%), FLAT
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from shared.config.settings import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

TARGET_UP_THRESHOLD = 0.005
TARGET_DOWN_THRESHOLD = -0.005
N_SPLITS = 5


def build_target(df: pd.DataFrame) -> pd.Series:
    future_return = df["close"].shift(-1) / df["close"] - 1
    target = pd.cut(
        future_return,
        bins=[-np.inf, TARGET_DOWN_THRESHOLD, TARGET_UP_THRESHOLD, np.inf],
        labels=["DOWN", "FLAT", "UP"],
    )
    return target


FEATURE_COLS = [
    "ema_9", "ema_20", "ema_50", "ema_200",
    "rsi", "macd", "macd_hist", "stochrsi_k", "stochrsi_d",
    "adx", "dmp", "dmn", "atr_pct", "bb_pct", "bb_width", "squeeze",
    "volume_ratio", "volume_spike", "cmf", "mfi",
    "candle_pattern_score", "ema_9_20_cross",
    "price_above_ema_50", "price_above_ema_200", "price_above_vwap",
    "hl_range", "roc_10", "cci_20", "williams_r",
    "gap_up", "gap_down", "pct_from_52w_high", "pct_from_52w_low",
]


class GradientBoostingModel:
    """
    Ensemble of XGBoost + LightGBM with walk-forward validation.
    MLflow tracks every experiment run automatically.
    """

    def __init__(self, model_name: str = "xgboost_lgbm_ensemble"):
        self._model_name = model_name
        self._xgb_model: Optional[xgb.XGBClassifier] = None
        self._lgb_model: Optional[lgb.LGBMClassifier] = None
        self._encoder = LabelEncoder()
        mlflow.set_tracking_uri(settings.ml.mlflow_tracking_uri)

    def train(self, df: pd.DataFrame) -> dict:
        df = df.dropna(subset=FEATURE_COLS)
        y_raw = build_target(df).dropna()
        df = df.loc[y_raw.index]
        X = df[FEATURE_COLS].astype(float)
        y = self._encoder.fit_transform(y_raw)

        with mlflow.start_run(run_name=f"{self._model_name}_train"):
            mlflow.log_params({
                "model": self._model_name,
                "n_features": len(FEATURE_COLS),
                "n_samples": len(X),
                "target_up_threshold": TARGET_UP_THRESHOLD,
            })

            tscv = TimeSeriesSplit(n_splits=N_SPLITS)
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                xgb_clf = xgb.XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="mlogloss",
                    tree_method="hist", device="cpu",
                )
                xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                lgb_clf = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, verbose=-1,
                )
                lgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

                xgb_proba = xgb_clf.predict_proba(X_val)
                lgb_proba = lgb_clf.predict_proba(X_val)
                ensemble_proba = (xgb_proba * 0.5) + (lgb_proba * 0.5)
                preds = ensemble_proba.argmax(axis=1)
                report = classification_report(y_val, preds, output_dict=True)
                fold_scores.append(report["accuracy"])
                log.info("Fold %d accuracy: %.4f", fold, report["accuracy"])

            mean_accuracy = float(np.mean(fold_scores))
            mlflow.log_metric("mean_cv_accuracy", mean_accuracy)
            log.info("Mean CV Accuracy: %.4f", mean_accuracy)

            # Final model on full data
            self._xgb_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss", tree_method="hist",
            )
            self._xgb_model.fit(X, y)

            self._lgb_model = lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
            )
            self._lgb_model.fit(X, y)

            mlflow.xgboost.log_model(self._xgb_model, "xgboost_model")
            mlflow.lightgbm.log_model(self._lgb_model, "lightgbm_model")

        return {"mean_cv_accuracy": mean_accuracy, "n_features": len(FEATURE_COLS)}

    def predict_proba(self, features: dict) -> dict[str, float]:
        if not self._xgb_model or not self._lgb_model:
            raise RuntimeError("Model not trained or loaded.")
        X = pd.DataFrame([features])[FEATURE_COLS].astype(float)
        xgb_p = self._xgb_model.predict_proba(X)[0]
        lgb_p = self._lgb_model.predict_proba(X)[0]
        ensemble = (xgb_p * 0.5 + lgb_p * 0.5)
        classes = list(self._encoder.classes_)
        return {cls: float(prob) for cls, prob in zip(classes, ensemble)}
