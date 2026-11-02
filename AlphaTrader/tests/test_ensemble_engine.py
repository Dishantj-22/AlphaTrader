import pytest
from services.signal_engine.ensemble.engine import EnsembleEngine, EnsembleInput
from shared.schemas.signal import SignalType


def _make_input(**overrides) -> EnsembleInput:
    defaults = dict(
        symbol="RELIANCE",
        gb_up_prob=0.5,
        gb_down_prob=0.2,
        lstm_up_prob=0.5,
        lstm_down_prob=0.2,
        sentiment_score=0.0,
        volume_score=0.5,
        candlestick_score=0.0,
        atr=15.0,
        current_price=3000.0,
        adx=25.0,
    )
    return EnsembleInput(**{**defaults, **overrides})


class TestEnsembleEngine:

    def setup_method(self):
        self.engine = EnsembleEngine()

    def test_strong_bullish_inputs_produce_buy(self):
        inp = _make_input(
            gb_up_prob=0.85, gb_down_prob=0.05,
            lstm_up_prob=0.80, lstm_down_prob=0.05,
            sentiment_score=0.9,
            volume_score=0.9,
            candlestick_score=0.75,
            adx=35.0,
        )
        result = self.engine.decide(inp)
        assert result.signal in (SignalType.BUY, SignalType.STRONG_BUY)
        assert result.confidence > 0.65

    def test_strong_bearish_inputs_produce_sell(self):
        inp = _make_input(
            gb_up_prob=0.05, gb_down_prob=0.85,
            lstm_up_prob=0.05, lstm_down_prob=0.85,
            sentiment_score=-0.9,
            volume_score=0.1,
            candlestick_score=-0.75,
            adx=35.0,
        )
        result = self.engine.decide(inp)
        assert result.signal in (SignalType.SELL, SignalType.STRONG_SELL)
        assert result.confidence < 0.35

    def test_mixed_signals_produce_hold(self):
        inp = _make_input(
            gb_up_prob=0.35, gb_down_prob=0.35,
            lstm_up_prob=0.35, lstm_down_prob=0.35,
            sentiment_score=0.0,
            volume_score=0.5,
            candlestick_score=0.0,
        )
        result = self.engine.decide(inp)
        assert result.signal == SignalType.HOLD

    def test_low_adx_downgrades_strong_signal(self):
        inp = _make_input(
            gb_up_prob=0.85, gb_down_prob=0.05,
            lstm_up_prob=0.85, lstm_down_prob=0.05,
            sentiment_score=0.9,
            volume_score=0.9,
            candlestick_score=0.75,
            adx=12.0,  # weak trend
        )
        result = self.engine.decide(inp)
        # Low ADX should not allow STRONG_BUY
        assert result.signal != SignalType.STRONG_BUY

    def test_confidence_bounded_0_to_1(self):
        for _ in range(20):
            import random
            inp = _make_input(
                gb_up_prob=random.random(),
                gb_down_prob=random.random(),
                sentiment_score=random.uniform(-1, 1),
                volume_score=random.random(),
                candlestick_score=random.uniform(-1, 1),
            )
            result = self.engine.decide(inp)
            assert 0.0 <= result.confidence <= 1.0

    def test_contributions_weights_sum_to_one(self):
        inp = _make_input()
        result = self.engine.decide(inp)
        total_weight = sum(c.weight for c in result.contributions)
        assert abs(total_weight - 1.0) < 1e-6
