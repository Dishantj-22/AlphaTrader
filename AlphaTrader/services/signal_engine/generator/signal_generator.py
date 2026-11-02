"""
Signal generator: orchestrates the full prediction pipeline per symbol.
Consumes feature events → runs inference → ensemble → risk check → publishes signal.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import redis

from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaConsumerClient, KafkaProducerClient
from shared.schemas.signal import TradingSignal, TradingStyle, SignalType
from services.ml_engine.inference.engine import get_inference_engine
from services.signal_engine.ensemble.engine import EnsembleEngine, EnsembleInput
from services.risk_engine.calculator.risk_engine import RiskEngine, RiskInput

log = logging.getLogger(__name__)
settings = get_settings()

DEFAULT_PORTFOLIO_VALUE = 1_000_000.0
SIGNAL_DEDUP_TTL_SECONDS = 300


class SignalGenerator:
    """
    End-to-end pipeline runner.
    One instance runs per deployment; each symbol's features trigger a prediction.
    """

    def __init__(self):
        self._producer = KafkaProducerClient()
        self._inference = get_inference_engine()
        self._ensemble = EnsembleEngine()
        self._risk = RiskEngine()
        self._redis = redis.from_url(settings.db.redis_url, decode_responses=True)

    def process_feature_event(self, feature_data: dict) -> Optional[TradingSignal]:
        symbol: str = feature_data.get("symbol", "UNKNOWN")

        dedup_key = f"signal_sent:{symbol}"
        if self._redis.get(dedup_key):
            return None

        # --- ML Inference ---
        gb_proba = self._inference.predict_gb(feature_data)
        lstm_proba = self._inference.predict_lstm(
            self._build_dummy_sequence(feature_data)
        )

        # --- Ensemble ---
        sentiment_score = float(self._redis.get(f"sentiment:{symbol}") or 0.0)
        volume_ratio = float(feature_data.get("volume_ratio", 1.0))
        volume_score = min(1.0, max(0.0, (volume_ratio - 0.5) / 2.0))
        pattern_score = float(feature_data.get("candle_pattern_score", 0.0))
        atr = float(feature_data.get("atr", 0.01))
        close = float(feature_data.get("close", 0.0))
        adx = float(feature_data.get("adx", 20.0))

        ensemble_input = EnsembleInput(
            symbol=symbol,
            gb_up_prob=gb_proba.get("UP", 0.33),
            gb_down_prob=gb_proba.get("DOWN", 0.33),
            lstm_up_prob=lstm_proba.get("UP", 0.33),
            lstm_down_prob=lstm_proba.get("DOWN", 0.33),
            sentiment_score=sentiment_score,
            volume_score=volume_score,
            candlestick_score=pattern_score,
            atr=atr,
            current_price=close,
            adx=adx,
        )
        ensemble_out = self._ensemble.decide(ensemble_input)

        if ensemble_out.signal == SignalType.HOLD:
            return None

        # --- Risk Management ---
        risk_out = self._risk.calculate(RiskInput(
            symbol=symbol,
            entry_price=close,
            atr=atr,
            signal_direction=ensemble_out.signal.value,
            portfolio_value=DEFAULT_PORTFOLIO_VALUE,
        ))

        if not risk_out.is_valid:
            log.info("Signal for %s rejected by risk engine: %s", symbol, risk_out.rejection_reason)
            return None

        # --- Build Signal ---
        holding_days = self._estimate_holding(ensemble_out.signal)
        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            symbol=symbol,
            exchange=feature_data.get("exchange", "NSE"),
            generated_at=datetime.utcnow(),
            trading_style=TradingStyle.BTST if holding_days <= 2 else TradingStyle.SWING,
            signal=ensemble_out.signal,
            confidence=round(ensemble_out.confidence, 4),
            entry_price=risk_out.entry_price,
            target_price=risk_out.target,
            stoploss_price=risk_out.stoploss,
            risk_reward_ratio=risk_out.risk_reward_ratio,
            holding_days=holding_days,
            position_size_pct=risk_out.position_size_pct,
            model_contributions=ensemble_out.contributions,
            technicals_score=round((ensemble_input.gb_up_prob - ensemble_input.gb_down_prob + 1) / 2, 4),
            sentiment_score=round((sentiment_score + 1) / 2, 4),
            volume_score=round(volume_score, 4),
            pattern_score=round((pattern_score + 1) / 2, 4),
        )

        self._producer.publish(settings.kafka.topic_signals, signal, key=symbol)
        self._redis.setex(dedup_key, SIGNAL_DEDUP_TTL_SECONDS, "1")
        log.info("Signal generated | symbol=%s signal=%s confidence=%.2f", symbol, signal.signal, signal.confidence)
        return signal

    @staticmethod
    def _estimate_holding(signal: SignalType) -> int:
        return {
            SignalType.STRONG_BUY: 3,
            SignalType.BUY: 2,
            SignalType.SELL: 2,
            SignalType.STRONG_SELL: 3,
        }.get(signal, 1)

    @staticmethod
    def _build_dummy_sequence(feature_data: dict):
        from services.ml_engine.models.lstm_model import INPUT_FEATURES, SEQUENCE_LEN
        row = np.array([float(feature_data.get(f, 0.0)) for f in INPUT_FEATURES])
        return np.tile(row, (SEQUENCE_LEN, 1))
