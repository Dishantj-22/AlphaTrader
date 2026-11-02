"""
Rule-based chart pattern detection engine.
Operates on a pandas OHLCV DataFrame (daily candles).
Each detector returns a score:  +1.0 (strong bullish) to -1.0 (strong bearish)
                                  0.0 = no pattern detected

Detected patterns:
  - Support & Resistance levels
  - Double Top / Double Bottom
  - Head & Shoulders / Inverse Head & Shoulders
  - Bull Flag / Bear Flag
  - Ascending Triangle / Descending Triangle / Symmetrical Triangle
  - Cup & Handle
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PatternResult:
    name: str
    score: float          # -1.0 to +1.0
    confidence: float     # 0.0 to 1.0
    description: str


class ChartPatternEngine:

    def __init__(self, lookback: int = 60):
        self.lookback = lookback   # candles to look back

    def detect_all(self, df: pd.DataFrame) -> tuple[float, list[PatternResult]]:
        """
        Run all detectors on the last `lookback` candles.
        Returns (composite_score, list_of_found_patterns).
        composite_score is the average of all found pattern scores.
        """
        window = df.tail(self.lookback).copy().reset_index(drop=True)
        if len(window) < 20:
            return 0.0, []

        detectors = [
            self._support_resistance,
            self._double_top_bottom,
            self._head_and_shoulders,
            self._bull_bear_flag,
            self._triangle,
            self._cup_and_handle,
        ]

        found: list[PatternResult] = []
        for fn in detectors:
            result = fn(window)
            if result and abs(result.score) > 0.1:
                found.append(result)

        if not found:
            return 0.0, []

        # Weight by confidence
        total_conf = sum(p.confidence for p in found)
        composite = sum(p.score * p.confidence for p in found) / total_conf
        return round(composite, 4), found

    # ── 1. Support & Resistance ───────────────────────────────

    def _support_resistance(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Detect if current price is near a key support or resistance level.
        Levels are identified as swing highs/lows with multiple touches.
        Zone is ATR% based so it works correctly for Rs.80 and Rs.130,000 stocks.
        """
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        n      = len(df)
        current_price = closes[-1]
        # Proportional zone: 0.5x ATR% of price — scales with stock price
        atr_pct = self._atr_pct(df)
        zone    = current_price * atr_pct * 0.5

        # Find swing highs and lows
        swing_highs, swing_lows = [], []
        for i in range(2, n - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])

        if not swing_highs and not swing_lows:
            return None

        # Find strongest resistance (most touches near that level)
        def count_touches(level, prices, tol):
            return sum(1 for p in prices if abs(p - level) <= tol)

        # Near support
        for level in swing_lows:
            if abs(current_price - level) <= zone:
                touches = count_touches(level, lows, zone)
                if touches >= 2:
                    conf = min(1.0, touches / 5)
                    return PatternResult(
                        name="Support Level",
                        score=0.65,
                        confidence=conf,
                        description=f"Price near support Rs.{level:.0f} ({touches} touches)"
                    )

        # Near resistance
        for level in swing_highs:
            if abs(current_price - level) <= zone:
                touches = count_touches(level, highs, zone)
                if touches >= 2:
                    conf = min(1.0, touches / 5)
                    return PatternResult(
                        name="Resistance Level",
                        score=-0.55,
                        confidence=conf,
                        description=f"Price near resistance Rs.{level:.0f} ({touches} touches)"
                    )

        # Breakout above resistance
        for level in swing_highs:
            if current_price > level and current_price - level <= zone * 2:
                return PatternResult(
                    name="Resistance Breakout",
                    score=0.80,
                    confidence=0.75,
                    description=f"Breakout above resistance Rs.{level:.0f}"
                )

        # Breakdown below support
        for level in swing_lows:
            if current_price < level and level - current_price <= zone * 2:
                return PatternResult(
                    name="Support Breakdown",
                    score=-0.80,
                    confidence=0.75,
                    description=f"Breakdown below support Rs.{level:.0f}"
                )

        return None

    # ── 2. Double Top / Double Bottom ─────────────────────────

    def _double_top_bottom(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Double Top:   two peaks at similar price, bearish reversal.
        Double Bottom: two troughs at similar price, bullish reversal.
        """
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        n      = len(df)
        # Proportional tolerance: 1.2% of price — works for all price levels
        atr_pct = self._atr_pct(df)
        tol     = closes[-1] * atr_pct * 1.2

        # Find swing highs
        peaks = [i for i in range(2, n - 2)
                 if highs[i] >= highs[i-1] and highs[i] >= highs[i-2]
                 and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]]

        # Find swing lows
        troughs = [i for i in range(2, n - 2)
                   if lows[i] <= lows[i-1] and lows[i] <= lows[i-2]
                   and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]]

        # Double Top
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if (p2 - p1) >= 5:   # at least 5 bars apart
                h1, h2 = highs[p1], highs[p2]
                if abs(h1 - h2) <= tol:
                    neckline = min(lows[p1:p2+1])
                    if closes[-1] < neckline:
                        conf = 1.0 - abs(h1 - h2) / tol
                        return PatternResult(
                            name="Double Top",
                            score=-0.85,
                            confidence=round(conf, 2),
                            description=f"Double Top at Rs.{h1:.0f} / Rs.{h2:.0f}, neckline Rs.{neckline:.0f}"
                        )

        # Double Bottom
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if (t2 - t1) >= 5:
                l1, l2 = lows[t1], lows[t2]
                if abs(l1 - l2) <= tol:
                    neckline = max(highs[t1:t2+1])
                    if closes[-1] > neckline:
                        conf = 1.0 - abs(l1 - l2) / tol
                        return PatternResult(
                            name="Double Bottom",
                            score=0.85,
                            confidence=round(conf, 2),
                            description=f"Double Bottom at Rs.{l1:.0f} / Rs.{l2:.0f}, neckline Rs.{neckline:.0f}"
                        )

        return None

    # ── 3. Head & Shoulders ───────────────────────────────────

    def _head_and_shoulders(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Head & Shoulders:         three peaks, middle highest. Bearish.
        Inverse Head & Shoulders: three troughs, middle lowest. Bullish.
        """
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        n      = len(df)

        peaks   = [i for i in range(2, n - 2)
                   if highs[i] > highs[i-1] and highs[i] > highs[i-2]
                   and highs[i] > highs[i+1] and highs[i] > highs[i+2]]
        troughs = [i for i in range(2, n - 2)
                   if lows[i] < lows[i-1] and lows[i] < lows[i-2]
                   and lows[i] < lows[i+1] and lows[i] < lows[i+2]]

        # Head & Shoulders (bearish)
        if len(peaks) >= 3:
            ls, head, rs = peaks[-3], peaks[-2], peaks[-1]
            h_ls, h_head, h_rs = highs[ls], highs[head], highs[rs]
            # Use % tolerance so MRF (130k) and IDFCFIRSTB (80) both work
            shoulder_tol = h_head * self._atr_pct(df) * 2.0
            if (h_head > h_ls and h_head > h_rs
                    and abs(h_ls - h_rs) <= shoulder_tol
                    and (head - ls) >= 5 and (rs - head) >= 5):
                neckline = np.mean([lows[ls:head].min(), lows[head:rs].min()])
                if closes[-1] < neckline:
                    return PatternResult(
                        name="Head & Shoulders",
                        score=-0.90,
                        confidence=0.80,
                        description=f"H&S: LS={h_ls:.0f} Head={h_head:.0f} RS={h_rs:.0f}, neckline={neckline:.0f}"
                    )

        # Inverse Head & Shoulders (bullish)
        if len(troughs) >= 3:
            ls, head, rs = troughs[-3], troughs[-2], troughs[-1]
            l_ls, l_head, l_rs = lows[ls], lows[head], lows[rs]
            shoulder_tol = abs(l_head) * self._atr_pct(df) * 2.0
            if (l_head < l_ls and l_head < l_rs
                    and abs(l_ls - l_rs) <= shoulder_tol
                    and (head - ls) >= 5 and (rs - head) >= 5):
                neckline = np.mean([highs[ls:head].max(), highs[head:rs].max()])
                if closes[-1] > neckline:
                    return PatternResult(
                        name="Inverse Head & Shoulders",
                        score=0.90,
                        confidence=0.80,
                        description=f"IH&S: LS={l_ls:.0f} Head={l_head:.0f} RS={l_rs:.0f}, neckline={neckline:.0f}"
                    )

        return None

    # ── 4. Bull Flag / Bear Flag ──────────────────────────────

    def _bull_bear_flag(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Bull Flag: sharp rally (pole), then tight downward consolidation (flag).
        Bear Flag: sharp drop (pole), then tight upward consolidation (flag).
        """
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        n      = len(df)
        if n < 15:
            return None

        # Look at last 20 bars: first 10 = pole, last 10 = flag
        pole_end   = n - 10
        pole_start = max(0, pole_end - 10)
        flag_bars  = closes[pole_end:]

        pole_move  = closes[pole_end - 1] - closes[pole_start]
        pole_pct   = pole_move / (closes[pole_start] + 1e-9) * 100
        flag_range = (max(flag_bars) - min(flag_bars)) / (closes[pole_end - 1] + 1e-9) * 100
        flag_slope = (flag_bars[-1] - flag_bars[0]) / (len(flag_bars) + 1e-9)

        # Bull Flag: pole up >= 5%, flag range <= 3%, slight downward slope
        if pole_pct >= 5 and flag_range <= 3.0 and flag_slope < 0:
            conf = min(1.0, pole_pct / 15)
            return PatternResult(
                name="Bull Flag",
                score=0.80,
                confidence=round(conf, 2),
                description=f"Bull Flag: pole +{pole_pct:.1f}%, flag range {flag_range:.1f}%"
            )

        # Bear Flag: pole down >= 5%, flag range <= 3%, slight upward slope
        if pole_pct <= -5 and flag_range <= 3.0 and flag_slope > 0:
            conf = min(1.0, abs(pole_pct) / 15)
            return PatternResult(
                name="Bear Flag",
                score=-0.80,
                confidence=round(conf, 2),
                description=f"Bear Flag: pole {pole_pct:.1f}%, flag range {flag_range:.1f}%"
            )

        return None

    # ── 5. Triangle Patterns ──────────────────────────────────

    def _triangle(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Ascending Triangle:    flat resistance + rising support  → bullish
        Descending Triangle:   flat support + falling resistance → bearish
        Symmetrical Triangle:  converging highs & lows          → breakout direction
        """
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        n      = len(df)
        if n < 20:
            return None

        window = 20
        recent_highs = highs[-window:]
        recent_lows  = lows[-window:]
        x            = np.arange(window)

        # Fit linear regression to highs and lows
        high_slope = np.polyfit(x, recent_highs, 1)[0]
        low_slope  = np.polyfit(x, recent_lows,  1)[0]

        # Use ATR% for flat threshold so it scales with stock price
        atr_pct     = self._atr_pct(df)
        price       = closes[-1]
        flat_thresh = price * atr_pct * 0.05   # slope within 5% of daily ATR = flat

        is_high_flat  = abs(high_slope) < flat_thresh
        is_low_flat   = abs(low_slope)  < flat_thresh
        is_high_down  = high_slope < -flat_thresh
        is_low_up     = low_slope  >  flat_thresh
        is_converging = is_high_down and is_low_up

        # Ascending Triangle
        if is_high_flat and is_low_up:
            resistance = np.mean(recent_highs[-5:])
            if closes[-1] > resistance:
                return PatternResult(
                    name="Ascending Triangle Breakout",
                    score=0.82,
                    confidence=0.75,
                    description=f"Ascending triangle breakout above Rs.{resistance:.0f}"
                )
            return PatternResult(
                name="Ascending Triangle",
                score=0.55,
                confidence=0.60,
                description=f"Ascending triangle forming, resistance Rs.{resistance:.0f}"
            )

        # Descending Triangle
        if is_low_flat and is_high_down:
            support = np.mean(recent_lows[-5:])
            if closes[-1] < support:
                return PatternResult(
                    name="Descending Triangle Breakdown",
                    score=-0.82,
                    confidence=0.75,
                    description=f"Descending triangle breakdown below Rs.{support:.0f}"
                )
            return PatternResult(
                name="Descending Triangle",
                score=-0.55,
                confidence=0.60,
                description=f"Descending triangle forming, support Rs.{support:.0f}"
            )

        # Symmetrical Triangle — determine breakout direction
        if is_converging:
            if closes[-1] > recent_highs[-1]:
                return PatternResult(
                    name="Symmetrical Triangle Breakout UP",
                    score=0.70,
                    confidence=0.65,
                    description="Symmetrical triangle bullish breakout"
                )
            if closes[-1] < recent_lows[-1]:
                return PatternResult(
                    name="Symmetrical Triangle Breakout DOWN",
                    score=-0.70,
                    confidence=0.65,
                    description="Symmetrical triangle bearish breakdown"
                )
            return PatternResult(
                name="Symmetrical Triangle",
                score=0.0,
                confidence=0.50,
                description="Symmetrical triangle — awaiting breakout direction"
            )

        return None

    # ── 6. Cup & Handle ───────────────────────────────────────

    def _cup_and_handle(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """
        Cup & Handle: U-shaped price recovery followed by a small pullback (handle).
        Strong bullish continuation pattern.
        """
        closes = df["close"].values
        n      = len(df)
        if n < 40:
            return None

        # Cup: look at bars [0 to -15] for U-shape
        cup   = closes[:n - 10]
        handle = closes[n - 10:]

        cup_left  = cup[:len(cup) // 3]
        cup_mid   = cup[len(cup) // 3: 2 * len(cup) // 3]
        cup_right = cup[2 * len(cup) // 3:]

        left_avg  = np.mean(cup_left)
        mid_avg   = np.mean(cup_mid)
        right_avg = np.mean(cup_right)

        # U-shape: left and right similar, middle lower
        is_cup = (mid_avg < left_avg * 0.97
                  and mid_avg < right_avg * 0.97
                  and abs(left_avg - right_avg) / (left_avg + 1e-9) < 0.05)

        if not is_cup:
            return None

        # Handle: small pullback (< 5%) after cup
        handle_drop = (max(handle) - min(handle)) / (max(handle) + 1e-9) * 100
        if handle_drop > 8:
            return None

        # Breakout: current price above cup right rim
        rim = np.mean(cup_right)
        if closes[-1] >= rim:
            return PatternResult(
                name="Cup & Handle",
                score=0.88,
                confidence=0.78,
                description=f"Cup & Handle breakout above rim Rs.{rim:.0f}"
            )

        return None

    # ── Helper ─────────────────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(abs(h[1:] - c[:-1]),
                        abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))

    @staticmethod
    def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
        """
        ATR as a percentage of current price.
        Used for tolerance zones so patterns work correctly across all price ranges:
        Rs.80 stocks and Rs.130,000 stocks (MRF) get proportionally sized zones.
        """
        price = float(df["close"].iloc[-1])
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(abs(h[1:] - c[:-1]),
                        abs(l[1:] - c[:-1])))
        atr = float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))
        return atr / price if price > 0 else 0.015   # fallback 1.5%
