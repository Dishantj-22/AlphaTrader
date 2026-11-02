"""
Weighted ensemble decision engine.
Combines predictions from technical, news, volume, and pattern models
into a single Buy/Sell/Hold signal with a confidence score.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from shared.schemas.signal import SignalType, ModelContribution

log = logging.getLogger(__name__)

# Model weights — must sum to 1.0
ENSEMBLE_WEIGHTS = {
    "technical_gb": 0.40,
    "technical_lstm": 0.00,  # blended into technical via TechnicalEnsemble
    "news_sentiment": 0.25,
    "volume": 0.15,
    "candlestick_pattern": 0.20,
}

SIGNAL_THRESHOLDS = {
    SignalType.STRONG_BUY: 0.80,
    SignalType.BUY: 0.65,
    SignalType.HOLD: 0.35,
    SignalType.SELL: 0.20,
    SignalType.STRONG_SELL: 0.0,
}


@dataclass
class EnsembleInput:
    symbol: str
    # Probabilities from each model (UP probability 0-1)
    gb_up_prob: float
    gb_down_prob: float
    lstm_up_prob: float
    lstm_down_prob: float
    sentiment_score: float      # -1 to +1
    volume_score: float         # 0 to 1 (bullish volume pressure)
    candlestick_score: float    # -1 to +1 (pattern engine output)
    atr: float
    current_price: float
    adx: float


@dataclass
class EnsembleOutput:
    signal: SignalType
    confidence: float
    bullish_score: float
    bearish_score: float
    contributions: list[ModelContribution]


class EnsembleEngine:

    def decide(self, inp: EnsembleInput) -> EnsembleOutput:
        # --- Technical score (blend GB + LSTM 60/40) ---
        tech_bull = inp.gb_up_prob * 0.6 + inp.lstm_up_prob * 0.4
        tech_bear = inp.gb_down_prob * 0.6 + inp.lstm_down_prob * 0.4
        tech_score = tech_bull - tech_bear  # -1 to +1

        # --- Sentiment score (already -1 to +1) ---
        sentiment_norm = max(-1.0, min(1.0, inp.sentiment_score))

        # --- Volume score normalised to -1 to +1 ---
        volume_norm = (inp.volume_score * 2.0) - 1.0

        # --- Candlestick score (already -1 to +1) ---
        pattern_norm = max(-1.0, min(1.0, inp.candlestick_score))

        # Weighted composite score — range -1 to +1
        composite = (
            tech_score * ENSEMBLE_WEIGHTS["technical_gb"]
            + sentiment_norm * ENSEMBLE_WEIGHTS["news_sentiment"]
            + volume_norm * ENSEMBLE_WEIGHTS["volume"]
            + pattern_norm * ENSEMBLE_WEIGHTS["candlestick_pattern"]
        )

        # Normalise to 0-1 confidence
        confidence = (composite + 1.0) / 2.0

        signal = self._classify_signal(confidence, inp.adx)

        contributions = [
            ModelContribution(
                model_name="gradient_boosting",
                signal=SignalType.BUY if tech_score > 0 else SignalType.SELL,
                confidence=abs(tech_score),
                weight=ENSEMBLE_WEIGHTS["technical_gb"],
            ),
            ModelContribution(
                model_name="finbert_sentiment",
                signal=SignalType.BUY if sentiment_norm > 0 else (SignalType.SELL if sentiment_norm < 0 else SignalType.HOLD),
                confidence=abs(sentiment_norm),
                weight=ENSEMBLE_WEIGHTS["news_sentiment"],
            ),
            ModelContribution(
                model_name="volume_analysis",
                signal=SignalType.BUY if volume_norm > 0 else SignalType.SELL,
                confidence=abs(volume_norm),
                weight=ENSEMBLE_WEIGHTS["volume"],
            ),
            ModelContribution(
                model_name="candlestick_patterns",
                signal=SignalType.BUY if pattern_norm > 0 else (SignalType.SELL if pattern_norm < 0 else SignalType.HOLD),
                confidence=abs(pattern_norm),
                weight=ENSEMBLE_WEIGHTS["candlestick_pattern"],
            ),
        ]

        return EnsembleOutput(
            signal=signal,
            confidence=confidence,
            bullish_score=(composite + 1) / 2,
            bearish_score=(1 - (composite + 1) / 2),
            contributions=contributions,
        )

    @staticmethod
    def _classify_signal(confidence: float, adx: float) -> SignalType:
        # Low ADX = weak trend → downgrade strong signals to neutral
        trend_weak = adx < 20

        if confidence >= SIGNAL_THRESHOLDS[SignalType.STRONG_BUY]:
            return SignalType.BUY if trend_weak else SignalType.STRONG_BUY
        if confidence >= SIGNAL_THRESHOLDS[SignalType.BUY]:
            return SignalType.HOLD if trend_weak else SignalType.BUY
        if confidence >= SIGNAL_THRESHOLDS[SignalType.HOLD]:
            return SignalType.HOLD
        if confidence >= SIGNAL_THRESHOLDS[SignalType.SELL]:
            return SignalType.SELL if not trend_weak else SignalType.HOLD
        return SignalType.STRONG_SELL if not trend_weak else SignalType.SELL
