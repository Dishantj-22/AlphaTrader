"""
AlphaTrader -- Standalone Prediction CLI
No broker API. No Kafka. No database. Just predictions.

Usage:
    python predict.py --symbol RELIANCE
    python predict.py --symbol TCS --style swing
    python predict.py --symbol INFY HDFCBANK SBIN --days 365
    python predict.py --symbol all
"""

import sys
import io
import os
import argparse
import logging
import warnings
from pathlib import Path
from types import ModuleType
from datetime import datetime, timedelta

# ── Compatibility shim: stub numba for Python >= 3.14 ─────────
# numba does not yet support Python 3.14. pandas-ta imports it
# unconditionally, so we inject a lightweight stub that makes
# @njit a no-op pass-through decorator. All pandas-ta indicators
# still run correctly via pure NumPy/pandas paths.
if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401 — use the real one if available
    except ImportError:
        _numba_stub = ModuleType("numba")
        def _njit(*args, **kwargs):
            return args[0] if args and callable(args[0]) else (lambda f: f)
        _numba_stub.njit = _njit
        sys.modules["numba"] = _numba_stub

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from services.ingestion.adapters.provider_registry import get_provider, REGISTRY
from services.feature_engine.patterns.chart_patterns import ChartPatternEngine as _CPEngine
from services.signal_engine.market_regime import get_market_context, Regime
from services.signal_engine.dynamic_weights import get_dynamic_weights, save_signal
from services.ingestion.adapters.fii_dii_options import fetch_fii_dii, fetch_options_data
from services.risk_engine.earnings_filter import is_safe_to_trade
from services.news_sentiment.local_sentiment import get_sentiment_for_predict
from ml.training.model_loader import ml_predict_proba, is_model_available

_chart_engine = _CPEngine(lookback=60)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("predict")

# ── Index Symbol Lists ────────────────────────────────────────

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "WIPRO", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "NESTLEIND", "POWERGRID", "NTPC", "BAJAJFINSV", "TECHM",
    "ONGC", "COALINDIA", "DIVISLAB", "DRREDDY", "CIPLA",
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BPCL", "BRITANNIA",
    "EICHERMOT", "GRASIM", "HEROMOTOCO", "HINDALCO", "INDUSINDBK",
    "JSWSTEEL", "M&M", "PIDILITIND", "SBILIFE", "SHRIRAMFIN",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "HDFCLIFE", "BAJAJFINSV",
]

NIFTY_MIDCAP = [
    # Banking & Finance
    "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "MUTHOOTFIN", "CHOLAFIN",
    "L&TFH", "MANAPPURAM", "SUNDARMFIN",
    # IT & Tech
    "MPHASIS", "COFORGE", "PERSISTENT", "LTTS", "TATAELXSI",
    # Pharma & Healthcare
    "AUROPHARMA", "LUPIN", "TORNTPHARM", "ALKEM", "GLAND",
    "MAXHEALTH", "FORTIS",
    # Auto & Auto Ancillaries
    "MOTHERSON", "BALKRISIND", "MRF", "EXIDEIND", "BHARATFORG",
    "ASHOKLEY", "TVSMOTOR",
    # Chemicals & Specialty
    "PIDILITIND", "SRF", "DEEPAKNTR", "AAVAS", "TATACHEM",
    # Consumer & Retail
    "VOLTAS", "WHIRLPOOL", "HAVELLS", "CROMPTON", "AIAENG",
    "BERGEPAINT", "KANSAINER",
    # Infrastructure & Real Estate
    "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD", "GODREJPROP",
    # Metals & Mining
    "NATIONALUM", "HINDCOPPER", "SAIL", "NMDC",
    # Media & Entertainment
    "PVRINOX", "ZEEL",
    # Others
    "CONCOR", "GMRAIRPORT", "IRCTC", "POLICYBZR", "NAUKRI",
]

NIFTY_SMALLCAP = [
    # Finance & Microfinance
    "CREDITACC", "UJJIVANSFB", "EQUITASBNK", "SURYODAY", "ESAFSFB",
    "SPANDANA", "AAVAS",
    # IT & Tech
    "TANLA", "RATEGAIN", "ZOMATO", "PAYTM", "NYKAA",
    "CARTRADE", "EASEMYTRIP",
    # Pharma & Healthcare
    "GRANULES", "LAURUSLABS", "SOLARA", "SEQUENT", "MEDANTA",
    "KRSNAA", "VIJAYADIAG",
    # Consumer & Retail
    "VSTIND", "BAJAJELEC", "ORIENTELEC", "RAJESHEXPO", "TTKPRESTIG",
    "BATAINDIA", "RELAXO",
    # Auto & Ancillaries
    "ENDURANCE", "SUPRAJIT", "GABRIEL", "SUNDRMFAST", "LUMAXTECH",
    # Chemicals
    "NAVINFLUOR", "FINEORG", "ALKYLAMINE", "AARTI",
    # Infrastructure & Real Estate
    "MAHINDCIE", "HGINFRA", "KNR", "PNCINFRA", "AHLUCONT",
    # Textile
    "GRASIM", "RAYMOND", "ARVIND", "WELSPUNIND",
    # Metals
    "RATNAMANI", "JINDALSAW", "WELCORP",
    # Others
    "CDSL", "CAMS", "MFSL", "POLICYBZR", "IXIGO",
]

# Combined index (all 3 together)
NIFTY_ALL = list(dict.fromkeys(NIFTY50 + NIFTY_MIDCAP + NIFTY_SMALLCAP))

INDEX_MAP = {
    "all":      NIFTY50,
    "nifty50":  NIFTY50,
    "midcap":   NIFTY_MIDCAP,
    "smallcap": NIFTY_SMALLCAP,
    "full":     NIFTY_ALL,
}

def resolve_symbols(symbol_args: list[str]) -> tuple[list[str], str]:
    """
    Returns (symbol_list, label).
    Handles keywords: all, nifty50, midcap, smallcap, full
    """
    if len(symbol_args) == 1 and symbol_args[0].lower() in INDEX_MAP:
        key = symbol_args[0].lower()
        labels = {
            "all":      "Nifty 50 (50 stocks)",
            "nifty50":  "Nifty 50 (50 stocks)",
            "midcap":   "Nifty Midcap (50 stocks)",
            "smallcap": "Nifty Smallcap (50 stocks)",
            "full":     "Nifty 50 + Midcap + Smallcap (150 stocks)",
        }
        return INDEX_MAP[key], labels[key]
    return [s.upper() for s in symbol_args], f"{len(symbol_args)} stock(s)"


# ── 1. Data Fetching ──────────────────────────────────────────

def fetch_data(symbol: str, days: int = 365, provider_name: str | None = None) -> pd.DataFrame:
    provider = get_provider(provider_name)
    # Always fetch at least 400 days so EMA-200 / ADX have enough warmup bars.
    # The signal is generated from the last row only — extra history never hurts.
    fetch_days = max(days, 400)
    return provider.fetch(symbol, interval="day", days=fetch_days)


# ── 2. Feature Engineering ────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta
    d = df.copy()
    n = len(d)

    # ── Trend (4 indicators) ──────────────────────────────────
    for p in [9, 20, 50, 200]:
        d[f"ema_{p}"] = ta.ema(d["close"], length=p)

    # Supertrend — best single trend filter (price + volatility combined)
    st = ta.supertrend(d["high"], d["low"], d["close"], length=10, multiplier=3.0)
    d["supertrend_dir"] = st["SUPERTd_10_3.0"]

    # ADX — trend strength only (DI+/DI- dropped: duplicates EMA cross)
    adx_ = ta.adx(d["high"], d["low"], d["close"], length=14)
    d["adx"] = adx_["ADX_14"]

    # ── Momentum (2 indicators) ───────────────────────────────
    d["rsi"] = ta.rsi(d["close"], length=14)

    # MACD histogram only — captures momentum shift better than line
    # (MACD line dropped: duplicates EMA cross; ROC and CCI dropped: duplicate RSI/histogram)
    macd_       = ta.macd(d["close"], fast=12, slow=26, signal=9)
    d["macd_hist"] = macd_["MACDh_12_26_9"]

    # ── Volatility (2 indicators) ─────────────────────────────
    # ATR kept for risk calculations (stoploss / target)
    d["atr"]    = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["atr_pct"] = d["atr"] / d["close"] * 100

    # BB% — where price sits inside the band (width dropped: rarely actionable alone)
    bb_         = ta.bbands(d["close"], length=20, std=2.0)
    d["bb_upper"] = bb_["BBU_20_2.0_2.0"]
    d["bb_lower"] = bb_["BBL_20_2.0_2.0"]
    d["bb_mid"]   = bb_["BBM_20_2.0_2.0"]
    d["bb_pct"]   = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    # ── Volume (2 indicators) ─────────────────────────────────
    d["volume_sma20"] = d["volume"].rolling(20).mean()
    d["volume_ratio"] = d["volume"] / (d["volume_sma20"] + 1e-9)
    d["volume_spike"] = (d["volume_ratio"] > 2.0).astype(int)

    # CMF — best money flow indicator (OBV and MFI dropped: both correlated with CMF)
    d["cmf"] = ta.cmf(d["high"], d["low"], d["close"], d["volume"], length=20)

    # ── Structure (2 indicators) ──────────────────────────────
    # Gap detection — unique event signal, not captured by any oscillator
    d["gap_up"]   = ((d["open"] > d["close"].shift(1)) & (d["open"] > d["high"].shift(1))).astype(int)
    d["gap_down"] = ((d["open"] < d["close"].shift(1)) & (d["open"] < d["low"].shift(1))).astype(int)

    # 52-week position — contextual strength/weakness signal
    w52 = min(252, n - 1)
    d["52w_high"] = d["close"].rolling(w52).max()
    d["52w_low"]  = d["close"].rolling(w52).min()
    d["pct_from_52w_high"] = (d["close"] - d["52w_high"]) / (d["52w_high"] + 1e-9) * 100
    d["pct_from_52w_low"]  = (d["close"] - d["52w_low"])  / (d["52w_low"]  + 1e-9) * 100

    d.dropna(inplace=True)
    return d


# ── 3. Candlestick Patterns ───────────────────────────────────

def candlestick_score(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    c, p = df.iloc[-1], df.iloc[-2]
    body     = abs(c["close"] - c["open"])
    full_rng = c["high"] - c["low"] + 1e-9
    upper_w  = c["high"] - max(c["open"], c["close"])
    lower_w  = min(c["open"], c["close"]) - c["low"]
    is_bull  = c["close"] > c["open"]
    is_bear  = not is_bull
    p_bull   = p["close"] > p["open"]
    p_bear   = not p_bull

    if body / full_rng < 0.1:
        return 0.0  # Doji

    if is_bull and p_bear and c["open"] < p["close"] and c["close"] > p["open"]:
        return 0.75   # Bullish Engulfing
    if is_bear and p_bull and c["open"] > p["close"] and c["close"] < p["open"]:
        return -0.75  # Bearish Engulfing
    if body > 0 and lower_w / body >= 2.0 and upper_w / body < 0.3 and is_bull:
        return 0.60   # Hammer
    if body > 0 and upper_w / body >= 2.0 and lower_w / body < 0.3 and is_bear:
        return -0.60  # Shooting Star
    if is_bull and p_bear and c["open"] > p["close"] and c["close"] < p["open"]:
        return 0.40   # Bullish Harami
    if is_bear and p_bull and c["open"] < p["close"] and c["close"] > p["open"]:
        return -0.40  # Bearish Harami
    if is_bull and body / full_rng >= 0.92:
        return 0.65   # Marubozu Bullish
    if is_bear and body / full_rng >= 0.92:
        return -0.65  # Marubozu Bearish
    return 0.0


# ── 4. Heuristic ML Scoring ───────────────────────────────────

def technical_score(row: pd.Series) -> float:
    votes, weights = [], []

    def add(signal: bool, w: float):
        votes.append(1.0 if signal else 0.0)
        weights.append(w)

    # Trend — 45% weight
    add(row["close"] > row["ema_9"],          0.07)
    add(row["ema_9"]  > row["ema_20"],        0.08)
    add(row["ema_20"] > row["ema_50"],        0.07)
    add(row["close"] > row["ema_50"],         0.08)
    add(row["close"] > row["ema_200"],        0.10)
    add(row.get("supertrend_dir", 0) == 1,    0.05)

    # Momentum — 25% weight
    add(row["rsi"] > 50,                      0.10)
    add(30 < row["rsi"] < 70,                 0.05)
    add(row["macd_hist"] > 0,                 0.10)

    # Volatility — 10% weight
    add(0.2 < row["bb_pct"] < 0.8,           0.05)
    add(row["bb_pct"] > 0.5,                  0.05)

    # Volume — 10% weight
    add(row["volume_ratio"] > 1.0,            0.05)
    add(row["cmf"] > 0,                       0.05)

    # Context — 10% weight
    add(row["adx"] > 20,                      0.05)
    add(row["pct_from_52w_high"] > -10,       0.03)
    add(row["pct_from_52w_low"]  > 10,        0.02)

    total_w = sum(weights)
    return sum(v * w for v, w in zip(votes, weights)) / total_w


def volume_score(row: pd.Series) -> float:
    ratio    = min(row["volume_ratio"], 5.0) / 5.0
    cmf_norm = (row["cmf"] + 1) / 2.0
    return ratio * 0.55 + cmf_norm * 0.45


# ── Signal Confluence Score ───────────────────────────────────

def confluence_score(row: pd.Series, candle: float, chart_score: float,
                     weekly_trend: str | None, vol: float) -> dict:
    """
    Counts how many independent signals agree on direction.
    Returns a dict with bullish_count, bearish_count, total, and label.

    A high-confluence signal (6+/7 aligned) is far more reliable than 3/7.
    This is displayed in the output card and used to boost/reduce confidence.
    """
    bull_signals = []
    bear_signals = []

    # 1. EMA stack
    if row["ema_9"] > row["ema_20"] > row["ema_50"] and row["close"] > row["ema_200"]:
        bull_signals.append("EMA stack UP")
    elif row["ema_9"] < row["ema_20"] < row["ema_50"] and row["close"] < row["ema_200"]:
        bear_signals.append("EMA stack DOWN")

    # 2. Supertrend
    if row.get("supertrend_dir", 0) == 1:
        bull_signals.append("Supertrend UP")
    elif row.get("supertrend_dir", 0) == -1:
        bear_signals.append("Supertrend DOWN")

    # 3. RSI
    if row["rsi"] > 55:
        bull_signals.append(f"RSI {row['rsi']:.0f}")
    elif row["rsi"] < 45:
        bear_signals.append(f"RSI {row['rsi']:.0f}")

    # 4. MACD histogram
    if row["macd_hist"] > 0:
        bull_signals.append("MACD bullish")
    else:
        bear_signals.append("MACD bearish")

    # 5. Volume + CMF
    if vol > 0.60:
        bull_signals.append("Volume bullish")
    elif vol < 0.40:
        bear_signals.append("Volume bearish")

    # 6. Candlestick + Chart pattern combined
    pattern_score = (candle + chart_score) / 2
    if pattern_score > 0.3:
        bull_signals.append("Pattern bullish")
    elif pattern_score < -0.3:
        bear_signals.append("Pattern bearish")

    # 7. Weekly trend
    if weekly_trend == "UP":
        bull_signals.append("Weekly UP")
    elif weekly_trend == "DOWN":
        bear_signals.append("Weekly DOWN")

    total     = len(bull_signals) + len(bear_signals)
    bull_cnt  = len(bull_signals)
    bear_cnt  = len(bear_signals)

    if bull_cnt >= 6:
        label = "VERY HIGH (all aligned bullish)"
    elif bull_cnt >= 5:
        label = "HIGH"
    elif bull_cnt >= 4:
        label = "MODERATE"
    elif bear_cnt >= 6:
        label = "VERY HIGH (all aligned bearish)"
    elif bear_cnt >= 5:
        label = "HIGH bearish"
    elif bear_cnt >= 4:
        label = "MODERATE bearish"
    else:
        label = "LOW (mixed signals)"

    return {
        "bull":   bull_cnt,
        "bear":   bear_cnt,
        "total":  total,
        "label":  label,
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
    }


# ── 5. Ensemble ───────────────────────────────────────────────

# Default weights — overridden by dynamic weights after 20+ tracked trades
DEFAULT_WEIGHTS = {
    "technical":     0.35,
    "chart_pattern": 0.20,
    "candlestick":   0.10,
    "volume":        0.15,
    "sentiment":     0.20,
}

SIGNAL_THRESHOLDS = [
    (0.80, "STRONG BUY",  "[++] "),
    (0.65, "BUY",         "[ +] "),
    (0.55, "WEAK BUY",    "[ ~] "),
    (0.45, "HOLD",        "[ =] "),
    (0.35, "WEAK SELL",   "[ v] "),
    (0.20, "SELL",        "[ -] "),
    (0.00, "STRONG SELL", "[--] "),
]


def ensemble_confidence(tech: float, vol: float, candle: float,
                        chart: float = 0.0, sentiment: float = 0.5,
                        weights: dict | None = None) -> float:
    w           = weights or DEFAULT_WEIGHTS
    candle_norm = (candle + 1) / 2.0
    chart_norm  = (chart  + 1) / 2.0
    return (
        tech          * w["technical"]
        + chart_norm  * w["chart_pattern"]
        + candle_norm * w["candlestick"]
        + vol         * w["volume"]
        + sentiment   * w["sentiment"]
    )


def classify_signal(confidence: float, adx: float) -> tuple[str, str]:
    if adx < 20:
        confidence = max(0.21, min(0.79, confidence))
    for threshold, label, icon in SIGNAL_THRESHOLDS:
        if confidence >= threshold:
            return label, icon
    return "STRONG SELL", "[--] "


# ── 6. Risk Levels ────────────────────────────────────────────

def compute_risk(close: float, atr: float, signal: str, adx: float = 25.0) -> dict:
    """
    Volatility-adjusted ATR multipliers.
    Strong trending stocks (ADX > 30) use tighter stops — trend is reliable.
    Choppy stocks (ADX < 20) use wider stops to avoid noise-triggered exits.
    """
    is_long = "BUY" in signal

    if adx >= 30:
        sl_mult  = 1.2   # tight — strong trend, price won't retrace much
        tgt_mult = 2.8
    elif adx >= 20:
        sl_mult  = 1.5   # standard
        tgt_mult = 3.0
    else:
        sl_mult  = 2.0   # wide — choppy market, need room to breathe
        tgt_mult = 3.5   # higher target compensates for wider stop

    sl  = close - atr * sl_mult  if is_long else close + atr * sl_mult
    tgt = close + atr * tgt_mult if is_long else close - atr * tgt_mult
    rr  = abs(tgt - close) / (abs(close - sl) + 1e-9)
    return {
        "entry":  round(close, 2),
        "target": round(tgt, 2),
        "sl":     round(sl, 2),
        "rr":     round(rr, 2),
    }


# ── 7. Multi-Timeframe Confirmation ──────────────────────────

def _weekly_trend(symbol: str, provider_name: str | None) -> str | None:
    """
    Fetch weekly candles and return 'UP', 'DOWN', or None.
    A daily BUY signal against weekly downtrend is downgraded.
    """
    try:
        provider = get_provider(provider_name)
        df_w = provider.fetch(symbol, interval="day", days=600)
        if df_w.empty or len(df_w) < 40:
            return None
        # Resample to weekly
        df_w = df_w.resample("W").agg({
            "open": "first", "high": "max",
            "low": "min",  "close": "last", "volume": "sum"
        }).dropna()
        if len(df_w) < 20:
            return None
        ema20w = df_w["close"].ewm(span=20, adjust=False).mean().iloc[-1]
        cur    = df_w["close"].iloc[-1]
        return "UP" if cur > ema20w else "DOWN"
    except Exception:
        return None


# ── 8. Main Predict Function ──────────────────────────────────

def predict(symbol: str, days: int = 365, style: str = "swing",
            provider_name: str | None = None,
            market_ctx=None,
            fii_dii=None,
            options=None,
            ensemble_weights: dict | None = None) -> dict | None:
    try:
        # ── Earnings / event filter ───────────────────────────
        safe, block_reason = is_safe_to_trade(symbol)
        if not safe:
            log.info("Signal blocked for %s: %s", symbol, block_reason)
            return {"symbol": symbol, "signal": "BLOCKED", "icon": "[!!] ",
                    "confidence": 0.0, "block_reason": block_reason,
                    "entry": 0.0, "target": 0.0, "stoploss": 0.0,
                    "rr": 0.0, "holding_days": 0, "adx": 0.0, "rsi": 0.0,
                    "atr_pct": 0.0, "volume_ratio": 0.0,
                    "scores": {"technical": 0, "volume": 0,
                               "candlestick": 0, "chart": 0, "sentiment": 50},
                    "chart_patterns": [], "model_used": "blocked",
                    "regime": "N/A", "sentiment_sources": [],
                    "intraday_note": None, "data_points": 0}

        df = fetch_data(symbol, days, provider_name)
        df = compute_features(df)
        if len(df) < 5:
            log.warning("Not enough data for %s", symbol)
            return None

        row = df.iloc[-1]
        adx = float(row["adx"])

        # ── ML model or heuristic ────────────────────────────
        ml_proba = ml_predict_proba(row.to_dict())
        if ml_proba:
            buy_p = ml_proba.get("BUY", 0.33)
            sel_p = ml_proba.get("SELL", 0.33)
            tech  = float(np.clip(buy_p - sel_p + 0.5, 0.0, 1.0))
            model_used = "xgboost"
        else:
            tech  = technical_score(row)
            model_used = "heuristic"

        vol    = volume_score(row)
        candle = candlestick_score(df)
        chart_score, chart_patterns = _chart_engine.detect_all(df)

        # ── Weekly trend (needed for confluence + MTF) ────────
        weekly_trend = _weekly_trend(symbol, provider_name)

        # ── Confluence scoring ────────────────────────────────
        confluence = confluence_score(row, candle, chart_score, weekly_trend, vol)

        # ── Sentiment: News + FII/DII + Options PCR ──────────
        fii_sent  = fii_dii.sentiment  if fii_dii  else 0.0
        opt_sent  = options.sentiment  if options  else 0.0
        sentiment, news_sources = get_sentiment_for_predict(symbol, fii_sent, opt_sent)
        sentiment_sources = news_sources.copy()
        if fii_dii:
            sentiment_sources.append(f"FII/DII: {fii_dii.combined_net:+.0f}Cr")
        if options:
            sentiment_sources.append(f"PCR: {options.pcr:.2f}")

        # ── Dynamic weights ───────────────────────────────────
        w    = ensemble_weights or DEFAULT_WEIGHTS
        conf = ensemble_confidence(tech, vol, candle, chart=chart_score,
                                   sentiment=sentiment, weights=w)

        # ── Confluence boost / penalty ────────────────────────
        # High confluence (5+ aligned) boosts confidence by up to 8%
        # Low confluence (2 or fewer aligned) penalises by up to 8%
        bull_cnt = confluence["bull"]
        bear_cnt = confluence["bear"]
        is_bull_signal = conf > 0.50
        if is_bull_signal:
            if bull_cnt >= 5:
                conf = min(1.0, conf * 1.08)
            elif bull_cnt <= 2:
                conf = conf * 0.92
        else:
            if bear_cnt >= 5:
                conf = max(0.0, conf * 0.92)
            elif bear_cnt <= 2:
                conf = min(1.0, conf * 1.08)

        # ── Market regime: cap + min confidence ──────────────
        regime_label = "UNKNOWN"
        if market_ctx:
            regime_label = market_ctx.regime.value
            conf = min(conf, market_ctx.confidence_cap)
            if market_ctx.regime == Regime.BEAR and conf > 0.50:
                conf = min(conf, 0.49)
            if market_ctx.regime.value in ("CAUTION", "BEAR"):
                if conf < market_ctx.min_confidence and conf > 0.50:
                    conf = min(conf, 0.54)

        # ── Multi-timeframe confirmation ──────────────────────
        mtf_note = None
        if weekly_trend:
            if weekly_trend == "DOWN" and conf > 0.55:
                conf     = conf * 0.90
                mtf_note = "Weekly DOWN — BUY confidence reduced 10%"
            elif weekly_trend == "UP" and conf < 0.45:
                conf     = conf * 1.08
                mtf_note = "Weekly UP — SELL confidence reduced"
            else:
                mtf_note = f"Weekly {weekly_trend} — aligned"
            conf = float(np.clip(conf, 0.0, 1.0))

        # ── Time-of-day adjustment (intraday/btst only) ───────
        tod_mult, tod_note = _time_of_day_multiplier()
        if style in ("intraday", "btst") and tod_mult != 1.0:
            conf     = float(np.clip(conf * tod_mult, 0.0, 1.0))
            mtf_note = (mtf_note + " | " if mtf_note else "") + tod_note

        signal, icon = classify_signal(conf, adx)
        risk = compute_risk(float(row["close"]), float(row["atr"]), signal, adx)

        # ── Intraday momentum note ────────────────────────────
        intraday_note = None
        if style in ("intraday", "btst"):
            intraday_note = _intraday_momentum(symbol, provider_name)

        holding = {"intraday": 0, "btst": 1, "swing": 3}.get(style, 3)
        if "STRONG" in signal:
            holding = max(1, holding)

        result = {
            "symbol":            symbol,
            "signal":            signal,
            "icon":              icon,
            "confidence":        round(conf * 100, 1),
            "entry":             risk["entry"],
            "target":            risk["target"],
            "stoploss":          risk["sl"],
            "rr":                risk["rr"],
            "holding_days":      holding,
            "adx":               round(adx, 1),
            "rsi":               round(float(row["rsi"]), 1),
            "atr_pct":           round(float(row["atr_pct"]), 2),
            "volume_ratio":      round(float(row["volume_ratio"]), 2),
            "scores": {
                "technical":     round(tech * 100, 1),
                "volume":        round(vol * 100, 1),
                "candlestick":   round((candle + 1) / 2 * 100, 1),
                "chart":         round((chart_score + 1) / 2 * 100, 1),
                "sentiment":     round(sentiment * 100, 1),
            },
            "confluence":        confluence,
            "chart_patterns":    [{"name": p.name, "score": p.score,
                                    "desc": p.description} for p in chart_patterns],
            "model_used":        model_used,
            "regime":            regime_label,
            "weekly_trend":      weekly_trend or "N/A",
            "mtf_note":          mtf_note,
            "sentiment_sources": sentiment_sources,
            "intraday_note":     intraday_note,
            "data_points":       len(df),
            "block_reason":      None,
        }

        # ── Save signal for dynamic weight learning ───────────
        if signal not in ("HOLD", "WEAK BUY", "WEAK SELL", "BLOCKED"):
            try:
                save_signal({
                    "symbol": symbol,
                    "date":   str(df.index[-1])[:10],
                    "signal": signal,
                    "confidence": result["confidence"],
                    "scores": result["scores"],
                })
            except Exception:
                pass

        return result

    except Exception as exc:
        log.error("Prediction failed for %s: %s", symbol, exc)
        return None


def _intraday_momentum(symbol: str, provider_name: str | None) -> str | None:
    """
    Fetch last 5 days of 15-min candles and detect intraday trend.
    Returns None gracefully outside market hours instead of failing.
    """
    try:
        from datetime import timezone
        now_ist = datetime.now()
        hour    = now_ist.hour
        minute  = now_ist.minute

        # Outside NSE market hours (9:15 AM – 3:30 PM IST) — skip silently
        market_open  = hour > 9 or (hour == 9 and minute >= 15)
        market_close = hour < 15 or (hour == 15 and minute <= 30)
        if not (market_open and market_close):
            return "Market closed — intraday data not available"

        provider = get_provider(provider_name)
        df_15m   = provider.fetch(symbol, interval="15minute", days=5)
        if df_15m is None or df_15m.empty or len(df_15m) < 10:
            return None
        last_close = float(df_15m["close"].iloc[-1])
        prev_close = float(df_15m["close"].iloc[-6])
        chg        = (last_close - prev_close) / prev_close * 100
        vol_ratio  = float(df_15m["volume"].iloc[-1]) / (float(df_15m["volume"].mean()) + 1e-9)
        direction  = "UP" if chg > 0 else "DOWN"
        return f"Intraday {direction} {chg:+.1f}% (last 1.5h), vol {vol_ratio:.1f}x avg"
    except Exception:
        return None


def _time_of_day_multiplier() -> tuple[float, str]:
    """
    Returns (confidence_multiplier, note).
    Signal quality varies by time of day in Indian markets:
    - 9:15–9:30 AM : Opening volatility, unreliable signals → 0.85x
    - 9:30–11:00 AM: Morning trend establishment → 1.0x
    - 11:00–1:00 PM: Mid-session, most reliable → 1.05x
    - 1:00–2:00 PM : Lunch lull, low volume → 0.95x
    - 2:00–3:00 PM : Afternoon trend → 1.0x
    - 3:00–3:30 PM : Last-hour momentum, strong for BTST → 1.08x
    - Outside hours : End-of-day data, swing signals unaffected → 1.0x
    """
    now   = datetime.now()
    hour  = now.hour
    minute = now.minute
    t     = hour + minute / 60.0   # decimal hour

    if 9.25 <= t < 9.50:
        return 0.85, "Opening volatility (9:15-9:30) — signals less reliable"
    elif 9.50 <= t < 11.0:
        return 1.00, "Morning session (9:30-11:00)"
    elif 11.0 <= t < 13.0:
        return 1.05, "Mid-session (11:00-1:00) — most reliable"
    elif 13.0 <= t < 14.0:
        return 0.95, "Lunch lull (1:00-2:00) — lower volume"
    elif 14.0 <= t < 15.0:
        return 1.00, "Afternoon session (2:00-3:00)"
    elif 15.0 <= t < 15.50:
        return 1.08, "Last hour (3:00-3:30) — strong for BTST"
    else:
        return 1.00, "After hours — swing/BTST signals use end-of-day data"


# ── 8. Display ────────────────────────────────────────────────

SEP  = "+" + "-" * 51 + "+"
SEP2 = "+" + "-" * 51 + "+"


def print_result(r: dict) -> None:
    filled = int(r["confidence"] / 5)
    bar    = "#" * filled + "." * (20 - filled)
    w      = 51

    def row(left: str, right: str = "") -> str:
        content = f"  {left:<30}{right:>17}" if right else f"  {left}"
        return f"| {content:<{w}} |"

    print(SEP)
    print(f"| {r['icon']}{r['symbol']:<10}  Signal: {r['signal']:<15}{'':>4} |")
    print(SEP)
    print(row(f"Confidence  [{bar}]", f"{r['confidence']}%"))
    print(row(f"Entry       Rs. {r['entry']:>10,.2f}"))
    print(row(f"Target      Rs. {r['target']:>10,.2f}"))
    print(row(f"Stop Loss   Rs. {r['stoploss']:>10,.2f}"))
    print(row(f"R:R Ratio   {r['rr']:.2f}x"))
    print(row(f"Holding     {r['holding_days']} day(s)"))
    print(SEP)
    print(row(f"RSI    {r['rsi']:>5.1f}",    f"ADX       {r['adx']:>5.1f}"))
    print(row(f"ATR%   {r['atr_pct']:>5.2f}%", f"Vol Ratio {r['volume_ratio']:>5.2f}x"))
    print(SEP)
    print(row("Score Breakdown"))
    print(row(f"  Technical   : {r['scores']['technical']:>5.1f}%"))
    print(row(f"  Chart Pat.  : {r['scores']['chart']:>5.1f}%"))
    print(row(f"  Candlestick : {r['scores']['candlestick']:>5.1f}%"))
    print(row(f"  Volume      : {r['scores']['volume']:>5.1f}%"))
    print(row(f"  Sentiment   : {r['scores']['sentiment']:>5.1f}%"))
    print(row(f"  Model: {r.get('model_used','?'):<10} Regime: {r.get('regime','?')}"))
    weekly = r.get("weekly_trend", "N/A")
    print(row(f"  Weekly trend: {weekly}"))
    if r.get("mtf_note"):
        print(row(f"  {r['mtf_note']}"[:49]))
    if r.get("sentiment_sources"):
        print(row(f"  {' | '.join(r['sentiment_sources'])}"[:49]))
    # Confluence
    conf_data = r.get("confluence", {})
    if conf_data:
        bull = conf_data.get("bull", 0)
        bear = conf_data.get("bear", 0)
        label = conf_data.get("label", "")
        print(SEP)
        print(row(f"Signal Confluence"))
        print(row(f"  {bull} bullish / {bear} bearish signals aligned"))
        print(row(f"  {label}"[:49]))
        bull_sigs = conf_data.get("bull_signals", [])
        bear_sigs = conf_data.get("bear_signals", [])
        if bull_sigs:
            print(row(f"  [+] {', '.join(bull_sigs)}"[:49]))
        if bear_sigs:
            print(row(f"  [-] {', '.join(bear_sigs)}"[:49]))
    # Chart patterns found
    if r.get("chart_patterns"):
        print(SEP)
        print(row("Chart Patterns Detected"))
        for p in r["chart_patterns"]:
            direction = "[+]" if p["score"] > 0 else ("[-]" if p["score"] < 0 else "[=]")
            name_line = f"  {direction} {p['name']}"
            print(row(name_line[:49]))
            desc_line = f"      {p['desc']}"
            for chunk in [desc_line[i:i+47] for i in range(0, len(desc_line), 47)]:
                print(row(chunk))
    if r.get("intraday_note"):
        print(SEP)
        print(row(f"  {r['intraday_note']}"[:49]))
    print(row(f"  Data points : {r['data_points']}"))
    print(SEP)
    print()


def print_summary_table(results: list[dict]) -> None:
    buys  = [r for r in results if "BUY"  in r["signal"]]
    sells = [r for r in results if "SELL" in r["signal"]]
    holds = [r for r in results if r not in buys and r not in sells]

    w = 70
    print("\n" + "=" * w)
    print(f"  ALPHATRADER SIGNAL SUMMARY  ({len(results)} stocks scanned)")
    print("=" * w)
    print(f"  {'SYMBOL':<14} {'SIGNAL':<13} {'CONF':>6}  {'ENTRY':>9}  {'TARGET':>9}  {'SL':>9}  {'R:R':>4}")
    print("  " + "-" * 66)

    for r in sorted(results, key=lambda x: -x["confidence"]):
        print(
            f"  {r['icon']}{r['symbol']:<12} {r['signal']:<13} "
            f"{r['confidence']:>5.1f}%  "
            f"Rs.{r['entry']:>8,.0f}  "
            f"Rs.{r['target']:>8,.0f}  "
            f"Rs.{r['stoploss']:>8,.0f}  "
            f"{r['rr']:>4.1f}x"
        )

    print("=" * w)
    print(f"  BUY: {len(buys)}   SELL: {len(sells)}   HOLD/WEAK: {len(holds)}")
    print("=" * w)
    print("\n  [!] NOT financial advice. Always apply your own risk management.\n")


def print_top5(results: list[dict]) -> None:
    """
    After the main prediction, print the Top 5 Opportunities section:
      - Top 3 BUY picks  (highest confidence among BUY/STRONG BUY)
      - Top 2 SELL picks (highest confidence among SELL/STRONG SELL)
    Each card shows a compact trade setup with a one-line rationale.
    """
    if not results:
        return

    w = 70

    def _rationale(r: dict) -> str:
        parts = []
        if r["scores"]["technical"] >= 65:
            parts.append("strong trend")
        elif r["scores"]["technical"] >= 50:
            parts.append("mild trend")
        else:
            parts.append("weak trend")
        if r["scores"]["volume"] >= 65:
            parts.append("high volume")
        if r["adx"] >= 25:
            parts.append(f"ADX {r['adx']:.0f}")
        if r["rsi"] < 40:
            parts.append(f"RSI oversold {r['rsi']:.0f}")
        elif r["rsi"] > 60:
            parts.append(f"RSI overbought {r['rsi']:.0f}")
        if r["scores"]["candlestick"] >= 70:
            parts.append("bullish candle")
        elif r["scores"]["candlestick"] <= 30:
            parts.append("bearish candle")
        for p in r.get("chart_patterns", []):
            if abs(p["score"]) >= 0.7:
                parts.append(p["name"])
        return ", ".join(parts) if parts else "mixed signals"

    buy_picks  = sorted(
        [r for r in results if "BUY" in r["signal"]],
        key=lambda x: -x["confidence"]
    )[:3]

    sell_picks = sorted(
        [r for r in results if "SELL" in r["signal"]],
        key=lambda x: x["confidence"]   # lowest conf = strongest sell
    )[:2]

    if not buy_picks and not sell_picks:
        return

    print("\n" + "=" * w)
    print("  TOP 5 TRADE OPPORTUNITIES")
    print("=" * w)

    rank = 1

    if buy_picks:
        print(f"\n  -- TOP BUY PICKS --\n")
        for r in buy_picks:
            pnl_pct = (r["target"] - r["entry"]) / r["entry"] * 100
            print(f"  #{rank}  {r['icon']}{r['symbol']:<10}  {r['signal']:<13}  Confidence: {r['confidence']:.1f}%")
            print(f"      Entry  Rs. {r['entry']:>9,.2f}   Target Rs. {r['target']:>9,.2f}  (+{pnl_pct:.1f}%)")
            print(f"      SL     Rs. {r['stoploss']:>9,.2f}   R:R    {r['rr']:.2f}x   Hold: {r['holding_days']}d")
            print(f"      Why  : {_rationale(r)}")
            print()
            rank += 1

    if sell_picks:
        print(f"  -- TOP SELL / SHORT PICKS --\n")
        for r in sell_picks:
            pnl_pct = (r["entry"] - r["target"]) / r["entry"] * 100
            print(f"  #{rank}  {r['icon']}{r['symbol']:<10}  {r['signal']:<13}  Confidence: {r['confidence']:.1f}%")
            print(f"      Entry  Rs. {r['entry']:>9,.2f}   Target Rs. {r['target']:>9,.2f}  (-{pnl_pct:.1f}%)")
            print(f"      SL     Rs. {r['stoploss']:>9,.2f}   R:R    {r['rr']:.2f}x   Hold: {r['holding_days']}d")
            print(f"      Why  : {_rationale(r)}")
            print()
            rank += 1

    print("=" * w)
    print("  [!] NOT financial advice. Always apply your own risk management.")
    print("=" * w + "\n")


# ── 9. CLI Entry Point ────────────────────────────────────────

def main():
    provider_choices = list(REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="AlphaTrader -- NSE Stock Prediction CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Index keywords  (use as --symbol value):
  all        Nifty 50      (50 large-cap stocks)
  nifty50    Nifty 50      (same as 'all')
  midcap     Nifty Midcap  (50 mid-cap stocks)
  smallcap   Nifty Smallcap(50 small-cap stocks)
  full       All 3 combined (150 stocks — takes ~5 min)

Examples:
  python predict.py --symbol RELIANCE
  python predict.py --symbol TCS INFY SBIN --table
  python predict.py --symbol all --top5 --style btst
  python predict.py --symbol midcap --top5 --style btst
  python predict.py --symbol smallcap --top5 --style swing
  python predict.py --symbol full --table --style swing
  python predict.py --symbol HDFCBANK --provider zerodha

Provider setup (.env keys required):
  yfinance  -- no key needed (default)
  zerodha   -- ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN
  upstox    -- UPSTOX_ACCESS_TOKEN
  angel     -- ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
  groww     -- GROWW_AUTH_TOKEN  (from browser DevTools -> Authorization header)
        """
    )
    parser.add_argument(
        "--symbol", nargs="+", default=["RELIANCE"],
        help=(
            "NSE symbol(s) or index keyword.\n"
            "Keywords: all, nifty50, midcap, smallcap, full\n"
            "Examples: --symbol RELIANCE\n"
            "          --symbol TCS INFY WIPRO\n"
            "          --symbol midcap"
        ),
    )
    parser.add_argument("--days",     type=int, default=365,  help="Historical days (default: 365)")
    parser.add_argument("--style",    choices=["intraday", "btst", "swing"], default="swing")
    parser.add_argument("--provider", choices=provider_choices, default=None,
                        help=f"Data source. Auto-selects if omitted. Options: {', '.join(provider_choices)}")
    parser.add_argument("--table",    action="store_true", help="Compact summary table")
    parser.add_argument("--top5",     action="store_true",
                        help="Show Top 5 trade opportunities after predictions")
    args = parser.parse_args()

    # --top5 implies scanning enough stocks to find picks — default to Nifty 50
    if args.top5 and args.symbol == ["RELIANCE"]:
        args.symbol = ["all"]

    symbols, index_label = resolve_symbols(args.symbol)

    try:
        active_provider = get_provider(args.provider)
        provider_label  = active_provider.name.upper()
    except Exception as e:
        print(f"\n  [ERROR] Provider error: {e}\n")
        sys.exit(1)

    src_desc = {
        "YFINANCE": "Yahoo Finance (free, no API key)",
        "GROWW":    "Groww (unofficial API -- token from browser DevTools)",
    }.get(provider_label, f"{provider_label} broker API")

    print(f"\n  AlphaTrader | {index_label} | provider={provider_label} | style={args.style} | days={args.days}")
    print(f"  Data source : {src_desc}")

    # ── Fetch shared context once ─────────────────────────────
    print("  Fetching market context (Nifty/VIX/FII/Options)...", end=" ", flush=True)
    market_ctx = get_market_context()
    fii_dii    = fetch_fii_dii()
    options    = fetch_options_data()
    print("done")

    # ── Dynamic ensemble weights ──────────────────────────────
    dyn_weights = get_dynamic_weights()
    using_dynamic = dyn_weights != DEFAULT_WEIGHTS

    # ── Print market context header ───────────────────────────
    w = 70
    print("\n" + "=" * w)
    print("  MARKET CONTEXT")
    print("=" * w)
    regime_icons = {"BULL": "[^^]", "NEUTRAL": "[--]", "CAUTION": "[!-]", "BEAR": "[vv]"}
    regime_icon  = regime_icons.get(market_ctx.regime.value, "[??]")
    print(f"  {regime_icon} Regime  : {market_ctx.regime.value}  |  {market_ctx.message}")
    print(f"  Nifty 50  : Rs.{market_ctx.nifty_price:,.0f}  |  EMA200: Rs.{market_ctx.nifty_ema200:,.0f}  |  {market_ctx.pct_above_ema:+.1f}%")
    print(f"  India VIX : {market_ctx.india_vix:.1f}  |  Nifty RSI: {market_ctx.nifty_rsi:.1f}")
    if fii_dii:
        print(f"  FII/DII   : {fii_dii.message}")
    if options:
        print(f"  Options   : {options.message}")
    print(f"  Min conf  : {market_ctx.min_confidence*100:.0f}%  |  "
          f"Cap: {market_ctx.confidence_cap*100:.0f}%  |  "
          f"Size: {market_ctx.size_multiplier*100:.0f}% of normal")
    if market_ctx.regime.value == "BEAR":
        print(f"  [!] BEAR market — BUY signals suppressed")
    elif market_ctx.regime.value == "CAUTION":
        print(f"  [~] CAUTION — reduced size, higher confidence threshold")
    print("=" * w + "\n")

    # ── Model + weights notice ────────────────────────────────
    if is_model_available():
        print("  [ML] Trained XGBoost model loaded — using ML scoring")
    else:
        print("  [ML] No trained model found — using heuristic scoring")
        print("       Run: python ml/training/train_ml.py --symbol all --days 1825")
    if using_dynamic:
        print(f"  [W]  Dynamic weights active: {dyn_weights}")
    else:
        print(f"  [W]  Default weights (run 20+ trades to enable dynamic weighting)")
    print()

    results = []
    for sym in symbols:
        result = predict(sym, days=args.days, style=args.style,
                         provider_name=args.provider,
                         market_ctx=market_ctx,
                         fii_dii=fii_dii,
                         options=options,
                         ensemble_weights=dyn_weights)
        if result:
            # Skip BLOCKED signals from the main table/top5 but show them if verbose
            if result.get("signal") == "BLOCKED":
                if not args.table and not args.top5:
                    print(f"  [!!] {result['symbol']:<12} BLOCKED: {result.get('block_reason','')}")
                continue
            results.append(result)
            if not args.table and not args.top5:
                print_result(result)

    if args.table or len(symbols) > 1:
        print_summary_table(results)

    if args.top5:
        print_top5(results)


if __name__ == "__main__":
    main()
