"""
Correlation filter — prevents taking highly correlated positions.
Reads last N signals and rejects new ones if correlation with open
positions exceeds the threshold.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

MAX_CORRELATION   = 0.70   # reject if correlation > 70%
MAX_SECTOR_PCT    = 0.25   # max 25% of portfolio in one sector
LOOKBACK_DAYS     = 30     # days for correlation calculation

SECTOR_MAP = {
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "AXISBANK": "Banking",
    "KOTAKBANK": "Banking", "SBIN": "Banking", "INDUSINDBK": "Banking",
    "FEDERALBNK": "Banking", "IDFCFIRSTB": "Banking", "BANDHANBNK": "Banking",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT", "LTTS": "IT", "TATAELXSI": "IT",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "AUROPHARMA": "Pharma",
    "LUPIN": "Pharma", "TORNTPHARM": "Pharma",
    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto", "EICHERMOT": "Auto",
    "BAJAJFINSV": "Auto", "TVSMOTOR": "Auto", "ASHOKLEY": "Auto",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "COALINDIA": "Energy", "NTPC": "Energy", "POWERGRID": "Energy",
    # Metals
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "SAIL": "Metals", "NATIONALUM": "Metals", "HINDCOPPER": "Metals",
    # Finance / NBFC
    "BAJFINANCE": "Finance", "MUTHOOTFIN": "Finance", "CHOLAFIN": "Finance",
    "MANAPPURAM": "Finance", "HDFCLIFE": "Finance", "SBILIFE": "Finance",
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), "Other")


def check_correlation(
    new_symbol: str,
    open_symbols: list[str],
    days: int = LOOKBACK_DAYS,
) -> tuple[bool, str]:
    """
    Returns (is_allowed, reason).
    Fetches returns for new_symbol and all open_symbols,
    checks pairwise correlation.
    """
    if not open_symbols:
        return True, "No open positions"

    try:
        import yfinance as yf
        all_symbols = [new_symbol] + open_symbols
        tickers = [s + ".NS" for s in all_symbols]

        raw = yf.download(tickers, period=f"{days}d", interval="1d",
                          auto_adjust=True, progress=False, threads=False)

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(tickers)]
            close.columns = all_symbols
        else:
            close = raw[["close"]] if len(all_symbols) == 1 else raw
            close.columns = all_symbols

        returns = close.pct_change().dropna()
        if returns.empty or new_symbol not in returns.columns:
            return True, "Correlation check skipped (data unavailable)"

        new_ret = returns[new_symbol]
        for sym in open_symbols:
            if sym not in returns.columns:
                continue
            corr = float(new_ret.corr(returns[sym]))
            if corr > MAX_CORRELATION:
                return False, f"{new_symbol} is {corr:.0%} correlated with open position {sym} — skipping"

        return True, "Correlation OK"

    except Exception as exc:
        log.warning("Correlation check failed: %s — allowing trade", exc)
        return True, "Correlation check skipped (error)"


def check_sector_exposure(
    new_symbol: str,
    open_symbols: list[str],
    portfolio_value: float = 1_000_000,
    position_value: float = 45_000,
) -> tuple[bool, str]:
    """
    Checks if adding new_symbol would exceed sector concentration limit.
    """
    new_sector = get_sector(new_symbol)
    if new_sector == "Other":
        return True, "Sector not mapped — allowing"

    sector_symbols = [s for s in open_symbols if get_sector(s) == new_sector]
    current_exposure = len(sector_symbols) * position_value / portfolio_value
    new_exposure = current_exposure + (position_value / portfolio_value)

    if new_exposure > MAX_SECTOR_PCT:
        return False, (
            f"{new_sector} sector would be {new_exposure:.0%} of portfolio "
            f"(max {MAX_SECTOR_PCT:.0%}) — skipping {new_symbol}"
        )
    return True, f"{new_sector} sector exposure OK ({new_exposure:.0%})"
