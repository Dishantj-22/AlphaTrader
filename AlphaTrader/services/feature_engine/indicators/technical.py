"""
Technical indicator computation engine.
All indicators operate on pandas DataFrames with OHLCV columns.
Returns a single enriched DataFrame with 50+ derived features.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    ema_periods: list[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    vwap_enabled: bool = True

    def __post_init__(self):
        if self.ema_periods is None:
            self.ema_periods = [9, 20, 50, 200]


class TechnicalIndicatorEngine:

    def __init__(self, config: IndicatorConfig | None = None):
        self.config = config or IndicatorConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input df must have columns: open, high, low, close, volume
        Returns df with all indicators appended.
        """
        df = df.copy()
        df = self._trend_indicators(df)
        df = self._momentum_indicators(df)
        df = self._volatility_indicators(df)
        df = self._volume_indicators(df)
        df = self._market_structure(df)
        df.dropna(inplace=True)
        return df

    def _trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.config.ema_periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)

        supertrend = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        df["supertrend"] = supertrend["SUPERT_10_3.0"]
        df["supertrend_direction"] = supertrend["SUPERTd_10_3.0"]

        adx_data = ta.adx(df["high"], df["low"], df["close"], length=self.config.adx_period)
        df["adx"] = adx_data[f"ADX_{self.config.adx_period}"]
        df["dmp"] = adx_data[f"DMP_{self.config.adx_period}"]
        df["dmn"] = adx_data[f"DMN_{self.config.adx_period}"]

        df["ema_9_20_cross"] = np.where(df["ema_9"] > df["ema_20"], 1, -1)
        df["price_above_ema_50"] = (df["close"] > df["ema_50"]).astype(int)
        df["price_above_ema_200"] = (df["close"] > df["ema_200"]).astype(int)
        return df

    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ta.rsi(df["close"], length=self.config.rsi_period)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

        macd = ta.macd(
            df["close"],
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
        )
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]
        df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)

        stoch = ta.stochrsi(df["close"], length=self.config.stoch_k, rsi_length=self.config.rsi_period)
        df["stochrsi_k"] = stoch.iloc[:, 0]
        df["stochrsi_d"] = stoch.iloc[:, 1]

        df["roc_10"] = ta.roc(df["close"], length=10)
        df["momentum_10"] = ta.mom(df["close"], length=10)
        df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
        return df

    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.config.atr_period)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        bb = ta.bbands(df["close"], length=self.config.bb_period, std=self.config.bb_std)
        df["bb_upper"] = bb[f"BBU_{self.config.bb_period}_{self.config.bb_std}"]
        df["bb_mid"] = bb[f"BBM_{self.config.bb_period}_{self.config.bb_std}"]
        df["bb_lower"] = bb[f"BBL_{self.config.bb_period}_{self.config.bb_std}"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
        df["kc_upper"] = kc["KCUe_20_1.5"]
        df["kc_lower"] = kc["KCLe_20_1.5"]
        df["squeeze"] = ((df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])).astype(int)
        return df

    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

        if self.config.vwap_enabled and "vwap" not in df.columns:
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["price_above_vwap"] = (df["close"] > df.get("vwap", df["close"])).astype(int)
        return df

    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        df["daily_return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["gap_up"] = ((df["open"] > df["close"].shift(1)) & (df["open"] > df["high"].shift(1))).astype(int)
        df["gap_down"] = ((df["open"] < df["close"].shift(1)) & (df["open"] < df["low"].shift(1))).astype(int)
        df["52w_high"] = df["close"].rolling(252).max()
        df["52w_low"] = df["close"].rolling(252).min()
        df["pct_from_52w_high"] = (df["close"] - df["52w_high"]) / df["52w_high"] * 100
        df["pct_from_52w_low"] = (df["close"] - df["52w_low"]) / df["52w_low"] * 100
        return df
