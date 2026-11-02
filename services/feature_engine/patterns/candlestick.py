"""
Rule-based candlestick pattern recognition engine.
Each pattern returns a score: positive = bullish, negative = bearish, 0 = neutral.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class PatternBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class PatternResult:
    name: str
    bias: PatternBias
    score: float
    row_index: int


PATTERN_SCORES: dict[str, float] = {
    "hammer": 0.65,
    "inverted_hammer": 0.50,
    "bullish_engulfing": 0.75,
    "morning_star": 0.80,
    "bullish_harami": 0.45,
    "piercing_line": 0.55,
    "marubozu_bullish": 0.70,
    "three_white_soldiers": 0.85,
    "doji": 0.0,
    "spinning_top": 0.0,
    "shooting_star": -0.65,
    "hanging_man": -0.55,
    "bearish_engulfing": -0.75,
    "evening_star": -0.80,
    "bearish_harami": -0.45,
    "dark_cloud_cover": -0.55,
    "marubozu_bearish": -0.70,
    "three_black_crows": -0.85,
}


class CandlestickPatternEngine:

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns input df with an additional `candle_pattern_score` column
        representing the composite candlestick signal for each row.
        """
        scores = np.zeros(len(df))
        for i in range(2, len(df)):
            c = df.iloc[i]
            p = df.iloc[i - 1]
            pp = df.iloc[i - 2]
            detected = self._detect_at(c, p, pp)
            if detected:
                scores[i] = detected[0].score
        df = df.copy()
        df["candle_pattern_score"] = scores
        return df

    def _detect_at(self, c: pd.Series, p: pd.Series, pp: pd.Series) -> list[PatternResult]:
        patterns = []
        idx = c.name if hasattr(c, "name") else 0

        def add(name: str):
            score = PATTERN_SCORES.get(name, 0.0)
            bias = PatternBias.BULLISH if score > 0 else (PatternBias.BEARISH if score < 0 else PatternBias.NEUTRAL)
            patterns.append(PatternResult(name=name, bias=bias, score=score, row_index=idx))

        body = abs(c["close"] - c["open"])
        full_range = c["high"] - c["low"]
        upper_wick = c["high"] - max(c["open"], c["close"])
        lower_wick = min(c["open"], c["close"]) - c["low"]
        is_bullish = c["close"] > c["open"]
        is_bearish = c["close"] < c["open"]
        prev_body = abs(p["close"] - p["open"])
        prev_bullish = p["close"] > p["open"]
        prev_bearish = p["close"] < p["open"]

        # Doji
        if full_range > 0 and body / full_range < 0.1:
            add("doji")
            return patterns

        # Hammer / Shooting Star
        if body > 0:
            lower_ratio = lower_wick / body
            upper_ratio = upper_wick / body
            if lower_ratio >= 2.0 and upper_ratio < 0.3 and is_bullish:
                add("hammer")
            if upper_ratio >= 2.0 and lower_ratio < 0.3 and is_bearish:
                add("shooting_star")
            if lower_ratio >= 2.0 and upper_ratio < 0.3 and is_bearish:
                add("hanging_man")
            if upper_ratio >= 2.0 and lower_ratio < 0.3 and is_bullish:
                add("inverted_hammer")

        # Marubozu
        if full_range > 0 and body / full_range >= 0.95:
            add("marubozu_bullish" if is_bullish else "marubozu_bearish")

        # Engulfing
        if is_bullish and prev_bearish and c["open"] < p["close"] and c["close"] > p["open"]:
            add("bullish_engulfing")
        if is_bearish and prev_bullish and c["open"] > p["close"] and c["close"] < p["open"]:
            add("bearish_engulfing")

        # Harami
        if is_bullish and prev_bearish and c["open"] > p["close"] and c["close"] < p["open"]:
            add("bullish_harami")
        if is_bearish and prev_bullish and c["open"] < p["close"] and c["close"] > p["open"]:
            add("bearish_harami")

        # Morning Star
        pp_bearish = pp["close"] < pp["open"]
        if (
            pp_bearish
            and abs(p["close"] - p["open"]) / (p["high"] - p["low"] + 1e-9) < 0.3
            and is_bullish
            and c["close"] > (pp["open"] + pp["close"]) / 2
        ):
            add("morning_star")

        # Evening Star
        pp_bullish = pp["close"] > pp["open"]
        if (
            pp_bullish
            and abs(p["close"] - p["open"]) / (p["high"] - p["low"] + 1e-9) < 0.3
            and is_bearish
            and c["close"] < (pp["open"] + pp["close"]) / 2
        ):
            add("evening_star")

        return patterns
