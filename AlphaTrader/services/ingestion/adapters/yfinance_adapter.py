"""
Free market data source using yfinance.
No broker API key required.
Fetches OHLCV data for NSE stocks with .NS suffix.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# yfinance interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1d, 1wk, 1mo
# Note: intraday data (< 1d) only available for last 60 days
INTERVAL_MAP = {
    "1minute":  "1m",
    "5minute":  "5m",
    "15minute": "15m",
    "60minute": "60m",
    "day":      "1d",
}


def nse_ticker(symbol: str) -> str:
    """Convert plain NSE symbol to yfinance format: RELIANCE → RELIANCE.NS"""
    return symbol.upper() + ".NS"


def fetch_ohlcv(
    symbol: str,
    interval: str = "day",
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for an NSE stock.

    Returns DataFrame with columns: open, high, low, close, volume
    indexed by timestamp (UTC).
    """
    yf_interval = INTERVAL_MAP.get(interval, "1d")
    ticker = nse_ticker(symbol)

    # yfinance period string based on days requested
    if days <= 7:
        period = f"{days}d"
    elif days <= 60:
        period = f"{days}d"
    else:
        period = f"{min(days, 1825)}d"   # max 5 years

    log.info("Fetching %s | interval=%s period=%s", ticker, yf_interval, period)

    df = yf.download(
        ticker,
        period=period,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol or market hours.")

    # yfinance >= 0.2.x returns MultiIndex columns like (Price, Ticker) — flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(inplace=True)
    return df


def fetch_multiple(symbols: list[str], interval: str = "day", days: int = 365) -> dict[str, pd.DataFrame]:
    """Batch fetch for multiple symbols. Returns {symbol: DataFrame}."""
    tickers = [nse_ticker(s) for s in symbols]
    raw = yf.download(
        tickers,
        period=f"{days}d",
        interval=INTERVAL_MAP.get(interval, "1d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    result = {}
    for sym, ticker in zip(symbols, tickers):
        try:
            df = raw[ticker].copy()
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "timestamp"
            df.dropna(inplace=True)
            result[sym] = df
        except Exception as exc:
            log.warning("Could not extract data for %s: %s", sym, exc)
    return result
