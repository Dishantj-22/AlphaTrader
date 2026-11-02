import pytest
import pandas as pd
import numpy as np
from services.feature_engine.indicators.technical import TechnicalIndicatorEngine, IndicatorConfig


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 3000 + np.cumsum(np.random.randn(n) * 10)
    return pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.002),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close":  close,
        "volume": np.random.randint(100_000, 5_000_000, n),
    })


class TestTechnicalIndicatorEngine:

    def test_compute_returns_dataframe(self, sample_ohlcv):
        engine = TechnicalIndicatorEngine()
        result = engine.compute(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_all_ema_columns_present(self, sample_ohlcv):
        engine = TechnicalIndicatorEngine(IndicatorConfig(ema_periods=[9, 20, 50, 200]))
        result = engine.compute(sample_ohlcv)
        for period in [9, 20, 50, 200]:
            assert f"ema_{period}" in result.columns, f"Missing ema_{period}"

    def test_rsi_bounded(self, sample_ohlcv):
        engine = TechnicalIndicatorEngine()
        result = engine.compute(sample_ohlcv)
        assert result["rsi"].between(0, 100).all(), "RSI values outside [0, 100]"

    def test_bb_ordering(self, sample_ohlcv):
        engine = TechnicalIndicatorEngine()
        result = engine.compute(sample_ohlcv)
        assert (result["bb_upper"] >= result["bb_mid"]).all()
        assert (result["bb_mid"] >= result["bb_lower"]).all()

    def test_volume_ratio_positive(self, sample_ohlcv):
        engine = TechnicalIndicatorEngine()
        result = engine.compute(sample_ohlcv)
        assert (result["volume_ratio"] >= 0).all()

    def test_no_nan_after_compute(self, sample_ohlcv):
        engine = TechnicalIndicatorEngine()
        result = engine.compute(sample_ohlcv)
        nan_cols = result.columns[result.isna().any()].tolist()
        assert len(nan_cols) == 0, f"NaN columns after compute: {nan_cols}"

    def test_minimum_required_columns(self, sample_ohlcv):
        required = ["rsi", "macd", "adx", "atr", "obv", "volume_ratio", "bb_pct"]
        engine = TechnicalIndicatorEngine()
        result = engine.compute(sample_ohlcv)
        for col in required:
            assert col in result.columns, f"Missing required column: {col}"
