"""
ML Model Loader — loads the trained XGBoost model from disk.
Returns predict_proba dict compatible with predict.py.
Falls back to heuristic scoring if model not found.
"""

import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MODEL_DIR   = Path(__file__).parent / "models"
_MODEL_PATH  = _MODEL_DIR / "xgb_model.pkl"
_ENC_PATH    = _MODEL_DIR / "label_encoder.pkl"
_FEATS_PATH  = _MODEL_DIR / "feature_cols.pkl"

_model   = None
_encoder = None
_feature_cols = None
_loaded  = False


def _load():
    global _model, _encoder, _feature_cols, _loaded
    if _loaded:
        return
    if _MODEL_PATH.exists() and _ENC_PATH.exists() and _FEATS_PATH.exists():
        try:
            with open(_MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            with open(_ENC_PATH, "rb") as f:
                _encoder = pickle.load(f)
            with open(_FEATS_PATH, "rb") as f:
                _feature_cols = pickle.load(f)
            log.info("XGBoost model loaded from %s", _MODEL_PATH)
        except Exception as exc:
            log.warning("Could not load model: %s — using heuristic fallback", exc)
    else:
        log.info("No trained model found at %s — using heuristic scoring", _MODEL_PATH)
    _loaded = True


def ml_predict_proba(feature_row: dict) -> dict[str, float]:
    """
    Returns {"BUY": p, "HOLD": p, "SELL": p} probabilities.
    Falls back to None if model not available (caller uses heuristic).
    """
    _load()
    if _model is None or _feature_cols is None:
        return None   # signal to caller: use heuristic

    try:
        X = pd.DataFrame([feature_row])[_feature_cols].astype(float)
        proba = _model.predict_proba(X)[0]
        classes = list(_encoder.classes_)
        return {cls: float(p) for cls, p in zip(classes, proba)}
    except Exception as exc:
        log.warning("ML inference error: %s — falling back to heuristic", exc)
        return None


def is_model_available() -> bool:
    _load()
    return _model is not None
