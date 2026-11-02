"""
Model inference service.
Loads trained models from MLflow registry and serves real-time predictions.
Exposes a clean interface consumed by the signal engine.
"""

import logging
from functools import lru_cache
from typing import Optional
import numpy as np
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pytorch
from shared.config.settings import get_settings
from .gradient_boosting import GradientBoostingModel, FEATURE_COLS
from .lstm_model import LSTMModelTrainer, INPUT_FEATURES, SEQUENCE_LEN

log = logging.getLogger(__name__)
settings = get_settings()


class ModelInferenceEngine:
    """
    Unified inference gateway for all ML models.
    Handles model loading, versioning, and prediction routing.
    """

    def __init__(self):
        mlflow.set_tracking_uri(settings.ml.mlflow_tracking_uri)
        self._gb_model: Optional[GradientBoostingModel] = None
        self._lstm_trainer: Optional[LSTMModelTrainer] = None

    def load_models(self) -> None:
        log.info("Loading models from MLflow registry...")
        try:
            self._gb_model = GradientBoostingModel()
            self._gb_model._xgb_model = mlflow.xgboost.load_model("models:/xgboost_lgbm_ensemble/Production")
            self._gb_model._lgb_model = mlflow.lightgbm.load_model("models:/xgboost_lgbm_ensemble/Production")
            log.info("Gradient boosting models loaded.")
        except Exception as exc:
            log.warning("Could not load GB models from registry: %s", exc)

        try:
            self._lstm_trainer = LSTMModelTrainer()
            self._lstm_trainer._model = mlflow.pytorch.load_model("models:/lstm_predictor/Production")
            self._lstm_trainer._model.to(self._lstm_trainer._device)
            log.info("LSTM model loaded.")
        except Exception as exc:
            log.warning("Could not load LSTM model from registry: %s", exc)

    def predict_gb(self, feature_row: dict) -> dict[str, float]:
        if not self._gb_model or not self._gb_model._xgb_model:
            return {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}
        try:
            return self._gb_model.predict_proba(feature_row)
        except Exception as exc:
            log.exception("GB inference error: %s", exc)
            return {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}

    def predict_lstm(self, sequence: np.ndarray) -> dict[str, float]:
        if not self._lstm_trainer or not self._lstm_trainer._model:
            return {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}
        try:
            return self._lstm_trainer.predict_proba(sequence)
        except Exception as exc:
            log.exception("LSTM inference error: %s", exc)
            return {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}


@lru_cache(maxsize=1)
def get_inference_engine() -> ModelInferenceEngine:
    engine = ModelInferenceEngine()
    engine.load_models()
    return engine
