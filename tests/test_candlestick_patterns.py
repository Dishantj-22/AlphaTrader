import pytest
import pandas as pd
import numpy as np
from services.feature_engine.patterns.candlestick import CandlestickPatternEngine


def _make_candle(open_, close_, high_offset=0.01, low_offset=0.01) -> dict:
    high = max(open_, close_) * (1 + high_offset)
    low = min(open_, close_) * (1 - low_offset)
    return {"open": open_, "close": close_, "high": high, "low": low, "volume": 100_000}


def _build_df(candles: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(candles)


class TestCandlestickPatternEngine:

    def setup_method(self):
        self.engine = CandlestickPatternEngine()

    def test_bullish_engulfing_positive_score(self):
        candles = [
            _make_candle(100, 90),   # bearish
            _make_candle(100, 90),   # bearish (prev)
            _make_candle(88, 102),   # bullish engulfing
        ]
        df = _build_df(candles)
        result = self.engine.detect_all(df)
        assert result["candle_pattern_score"].iloc[-1] > 0

    def test_bearish_engulfing_negative_score(self):
        candles = [
            _make_candle(90, 100),   # bullish
            _make_candle(90, 100),   # bullish (prev)
            _make_candle(102, 88),   # bearish engulfing
        ]
        df = _build_df(candles)
        result = self.engine.detect_all(df)
        assert result["candle_pattern_score"].iloc[-1] < 0

    def test_neutral_for_insufficient_data(self):
        candles = [_make_candle(100, 105)]
        df = _build_df(candles)
        result = self.engine.detect_all(df)
        assert result["candle_pattern_score"].iloc[0] == 0.0

    def test_score_column_always_present(self):
        candles = [_make_candle(100, 105) for _ in range(5)]
        df = _build_df(candles)
        result = self.engine.detect_all(df)
        assert "candle_pattern_score" in result.columns

    def test_score_bounded(self):
        candles = [_make_candle(100, 105) for _ in range(10)]
        df = _build_df(candles)
        result = self.engine.detect_all(df)
        assert result["candle_pattern_score"].between(-1.0, 1.0).all()
