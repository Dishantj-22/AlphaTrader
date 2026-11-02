"""
Market Regime Detector — 4-regime VIX classification.
Fetches Nifty 50 index and India VIX from Yahoo Finance.

Regime classification (Indian market calibrated):
  BULL     — Nifty above EMA200, VIX < 14        → full size, normal threshold
  NEUTRAL  — Nifty near EMA200, VIX 14–18        → 75% size, slightly raised threshold
  CAUTION  — VIX 18–22 OR Nifty 2–5% below EMA  → 50% size, raised threshold
  BEAR     — VIX > 22 OR Nifty > 5% below EMA   → 25% size, only SELL signals
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class Regime(str, Enum):
    BULL    = "BULL"
    NEUTRAL = "NEUTRAL"
    CAUTION = "CAUTION"
    BEAR    = "BEAR"


@dataclass
class MarketContext:
    regime:          Regime
    nifty_price:     float
    nifty_ema200:    float
    nifty_rsi:       float
    india_vix:       float
    pct_above_ema:   float
    confidence_cap:  float
    size_multiplier: float
    min_confidence:  float    # minimum confidence to generate any signal
    message:         str


def get_market_context() -> MarketContext:
    try:
        import yfinance as yf
        import pandas_ta as ta
        import sys
        from types import ModuleType
        if "numba" not in sys.modules:
            try:
                import numba  # noqa
            except ImportError:
                _s = ModuleType("numba")
                _s.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
                sys.modules["numba"] = _s

        nifty_raw = yf.download("^NSEI", period="400d", interval="1d",
                                auto_adjust=True, progress=False, threads=False)
        vix_raw   = yf.download("^INDIAVIX", period="10d", interval="1d",
                                auto_adjust=True, progress=False, threads=False)

        for raw in [nifty_raw, vix_raw]:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

        nifty_close = nifty_raw["close"].dropna()
        ema200      = float(ta.ema(nifty_close, length=200).iloc[-1])
        rsi         = float(ta.rsi(nifty_close, length=14).iloc[-1])
        current     = float(nifty_close.iloc[-1])
        pct_above   = (current - ema200) / ema200 * 100

        india_vix   = float(vix_raw["close"].dropna().iloc[-1]) if not vix_raw.empty else 16.0

        # ── 4-Regime Classification (India VIX calibrated) ───
        if current > ema200 and india_vix < 14:
            regime   = Regime.BULL
            cap      = 1.0
            min_conf = 0.55
            size     = 1.0
            msg = f"BULL: Nifty {pct_above:+.1f}% above EMA200, VIX {india_vix:.1f} — trade normally at full size"

        elif india_vix > 22 or pct_above < -5:
            regime   = Regime.BEAR
            cap      = 0.65
            min_conf = 0.70    # only very high conviction signals pass
            size     = 0.25
            msg = f"BEAR: Nifty {pct_above:+.1f}% vs EMA200, VIX {india_vix:.1f} — only SELL signals, 25% size"

        elif india_vix > 18 or pct_above < -2:
            regime   = Regime.CAUTION
            cap      = 0.80
            min_conf = 0.65
            size     = 0.50
            msg = f"CAUTION: VIX {india_vix:.1f}, Nifty {pct_above:+.1f}% — reduced size, higher threshold"

        else:
            regime   = Regime.NEUTRAL
            cap      = 0.90
            min_conf = 0.60
            size     = 0.75
            msg = f"NEUTRAL: Nifty {pct_above:+.1f}% above EMA200, VIX {india_vix:.1f} — trade at 75% size"

        return MarketContext(
            regime=regime, nifty_price=current, nifty_ema200=ema200,
            nifty_rsi=rsi, india_vix=india_vix, pct_above_ema=round(pct_above, 2),
            confidence_cap=cap, size_multiplier=size, min_confidence=min_conf,
            message=msg,
        )

    except Exception as exc:
        log.warning("Market context unavailable: %s — defaulting to NEUTRAL", exc)
        return MarketContext(
            regime=Regime.NEUTRAL, nifty_price=0, nifty_ema200=0,
            nifty_rsi=50, india_vix=16, pct_above_ema=0,
            confidence_cap=0.90, size_multiplier=0.75, min_confidence=0.60,
            message="Market context unavailable — defaulting to NEUTRAL",
        )
