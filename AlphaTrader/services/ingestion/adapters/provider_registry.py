"""
Pluggable data source registry.

Default provider: yfinance (no API key needed).
Optional providers: zerodha, upstox, angel, groww — activated via .env or --provider flag.

Adding a new provider:
  1. Subclass DataProvider
  2. Implement fetch() and is_available()
  3. Register it in REGISTRY below
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ── Base contract ─────────────────────────────────────────────

class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if credentials/deps are present and valid."""
        ...

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        interval: str = "day",
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame indexed by timestamp.
        Columns: open, high, low, close, volume (all lowercase).
        """
        ...


# ── Provider 1: yfinance (free, default) ─────────────────────

class YFinanceProvider(DataProvider):
    name = "yfinance"

    INTERVAL_MAP = {
        "1minute":  "1m",
        "5minute":  "5m",
        "15minute": "15m",
        "60minute": "60m",
        "day":      "1d",
    }

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self, symbol: str, interval: str = "day", days: int = 365) -> pd.DataFrame:
        import yfinance as yf
        ticker = symbol.upper() + ".NS"
        yf_interval = self.INTERVAL_MAP.get(interval, "1d")
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval=yf_interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            raise ValueError(f"yfinance returned no data for {ticker}")
        # yfinance >= 0.2.x returns MultiIndex (Price, Ticker) — flatten to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]].dropna()


# ── Provider 2: Zerodha Kite Connect (optional) ───────────────

class ZerodhaProvider(DataProvider):
    name = "zerodha"

    INTERVAL_MAP = {
        "1minute":  "minute",
        "5minute":  "5minute",
        "15minute": "15minute",
        "60minute": "60minute",
        "day":      "day",
    }

    def is_available(self) -> bool:
        try:
            from kiteconnect import KiteConnect  # noqa: F401
            creds = self._load_creds()
            return bool(creds.get("api_key") and creds.get("access_token"))
        except Exception:
            return False

    def fetch(self, symbol: str, interval: str = "day", days: int = 365) -> pd.DataFrame:
        from kiteconnect import KiteConnect
        creds = self._load_creds()
        kite = KiteConnect(api_key=creds["api_key"])
        kite.set_access_token(creds["access_token"])

        instruments = kite.instruments("NSE")
        token_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
        token = token_map.get(symbol.upper())
        if not token:
            raise ValueError(f"Zerodha: symbol not found: {symbol}")

        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        raw = kite.historical_data(token, from_dt, to_dt, self.INTERVAL_MAP.get(interval, "day"))
        df = pd.DataFrame(raw)
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()

    @staticmethod
    def _load_creds() -> dict:
        import os
        return {
            "api_key":      os.getenv("ZERODHA_API_KEY", ""),
            "access_token": os.getenv("ZERODHA_ACCESS_TOKEN", ""),
        }


# ── Provider 3: Upstox v2 (optional) ─────────────────────────

class UpstoxProvider(DataProvider):
    name = "upstox"

    INTERVAL_MAP = {
        "1minute":  "1minute",
        "5minute":  "30minute",
        "15minute": "30minute",
        "60minute": "1hour",
        "day":      "1day",
    }

    def is_available(self) -> bool:
        try:
            import upstox_client  # noqa: F401
            import os
            return bool(os.getenv("UPSTOX_ACCESS_TOKEN"))
        except Exception:
            return False

    def fetch(self, symbol: str, interval: str = "day", days: int = 365) -> pd.DataFrame:
        import upstox_client
        import os
        from datetime import date

        configuration = upstox_client.Configuration()
        configuration.access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
        api_client = upstox_client.ApiClient(configuration)
        history_api = upstox_client.HistoryApi(api_client)

        instrument_key = f"NSE_EQ|{symbol.upper()}"
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days)).isoformat()

        resp = history_api.get_historical_candle_data1(
            instrument_key,
            self.INTERVAL_MAP.get(interval, "1day"),
            to_date,
            from_date,
            api_version="v2",
        )
        candles = resp.data.candles if resp.data else []
        if not candles:
            raise ValueError(f"Upstox returned no data for {symbol}")

        rows = [
            {
                "timestamp": c[0],
                "open":  c[1], "high": c[2],
                "low":   c[3], "close": c[4],
                "volume": c[5],
            }
            for c in candles
        ]
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].dropna()


# ── Provider 4: Angel One Smart API (optional) ────────────────

class AngelOneProvider(DataProvider):
    name = "angel"

    INTERVAL_MAP = {
        "1minute":  "ONE_MINUTE",
        "5minute":  "FIVE_MINUTE",
        "15minute": "FIFTEEN_MINUTE",
        "60minute": "ONE_HOUR",
        "day":      "ONE_DAY",
    }

    def is_available(self) -> bool:
        try:
            from SmartApi import SmartConnect  # noqa: F401
            import os
            return bool(os.getenv("ANGEL_API_KEY") and os.getenv("ANGEL_CLIENT_ID"))
        except Exception:
            return False

    def fetch(self, symbol: str, interval: str = "day", days: int = 365) -> pd.DataFrame:
        import os
        import pyotp
        from SmartApi import SmartConnect

        api_key   = os.getenv("ANGEL_API_KEY", "")
        client_id = os.getenv("ANGEL_CLIENT_ID", "")
        password  = os.getenv("ANGEL_PASSWORD", "")
        totp_key  = os.getenv("ANGEL_TOTP_SECRET", "")

        obj = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_key).now()
        obj.generateSession(client_id, password, totp)

        # Lookup token from symbol list
        import requests
        r = requests.get(
            "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
            timeout=10,
        )
        scripts = r.json()
        token = next(
            (s["token"] for s in scripts if s["name"] == symbol.upper() and s["exch_seg"] == "NSE"),
            None,
        )
        if not token:
            raise ValueError(f"Angel One: symbol not found: {symbol}")

        to_dt   = datetime.now().strftime("%Y-%m-%d %H:%M")
        from_dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")

        params = {
            "exchange":    "NSE",
            "symboltoken": token,
            "interval":    self.INTERVAL_MAP.get(interval, "ONE_DAY"),
            "fromdate":    from_dt,
            "todate":      to_dt,
        }
        resp = obj.getCandleData(params)
        candles = resp.get("data", [])
        if not candles:
            raise ValueError(f"Angel One returned no data for {symbol}")

        rows = [
            {
                "timestamp": c[0],
                "open": c[1], "high": c[2],
                "low":  c[3], "close": c[4],
                "volume": c[5],
            }
            for c in candles
        ]
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].dropna()


# ── Provider 5: Groww (optional) ─────────────────────────────
#
# Groww does not publish an official public trading API.
# However, their platform exposes an internal REST endpoint used
# by the Groww web app that returns historical OHLCV data.
#
# ⚠️  Important:
#   - This uses an unofficial endpoint (reverse-engineered from the web app).
#   - No API key is required — Groww uses session cookies + a user token.
#   - Set GROWW_AUTH_TOKEN in your .env (copy from browser DevTools →
#     Network tab → any Groww request → Authorization header).
#   - This may break if Groww updates their internal API.
#   - Use yfinance or a broker API for production reliability.

class GrowwProvider(DataProvider):
    name = "groww"

    # Groww internal API base (from web app network traffic)
    _BASE = "https://groww.in/v1/api/stocks_data/v1/accord_points/exchange/NSE/segment/CASH"

    INTERVAL_MAP = {
        "1minute":  "1",
        "5minute":  "5",
        "15minute": "15",
        "60minute": "60",
        "day":      "1D",
    }

    def is_available(self) -> bool:
        import os
        try:
            import requests  # noqa: F401
            return bool(os.getenv("GROWW_AUTH_TOKEN"))
        except ImportError:
            return False

    def fetch(self, symbol: str, interval: str = "day", days: int = 365) -> pd.DataFrame:
        import os
        import requests
        from datetime import date

        token = os.getenv("GROWW_AUTH_TOKEN", "")
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Origin": "https://groww.in",
            "Referer": "https://groww.in/stocks",
        }

        # Groww uses 'candleInterval' and date range params
        groww_interval = self.INTERVAL_MAP.get(interval, "1D")
        to_date   = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days)).isoformat()

        url = f"{self._BASE}/{symbol.upper()}/candles"
        params = {
            "startTimeInEpoch": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "endTimeInEpoch":   int(datetime.now().timestamp() * 1000),
            "intervalInMinutes": groww_interval,
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 401:
            raise RuntimeError(
                "Groww auth failed. Refresh GROWW_AUTH_TOKEN in .env.\n"
                "Get it from: browser DevTools → Network → any groww.in request → Authorization header."
            )
        resp.raise_for_status()

        data = resp.json()
        candles = data.get("candles") or data.get("data", {}).get("candles", [])
        if not candles:
            raise ValueError(f"Groww returned no candle data for {symbol}")

        rows = []
        for c in candles:
            # Groww candle format: [epoch_ms, open, high, low, close, volume]
            rows.append({
                "timestamp": pd.to_datetime(c[0], unit="ms", utc=True).tz_convert("Asia/Kolkata"),
                "open":   float(c[1]),
                "high":   float(c[2]),
                "low":    float(c[3]),
                "close":  float(c[4]),
                "volume": int(c[5]),
            })

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].dropna()


# ── Registry ──────────────────────────────────────────────────

REGISTRY: dict[str, DataProvider] = {
    p.name: p()
    for p in [YFinanceProvider, ZerodhaProvider, UpstoxProvider, AngelOneProvider, GrowwProvider]
}


def get_provider(name: Optional[str] = None) -> DataProvider:
    """
    Return the requested provider if available, else auto-select the best one.
    Priority: requested → zerodha → upstox → angel → groww → yfinance
    """
    if name:
        provider = REGISTRY.get(name.lower())
        if provider is None:
            raise ValueError(f"Unknown provider '{name}'. Choose from: {list(REGISTRY)}")
        if not provider.is_available():
            raise RuntimeError(
                f"Provider '{name}' is not available. "
                f"Install its package and set required env vars.\n"
                f"See .env.example for required keys."
            )
        return provider

    # Auto-select: prefer official broker APIs, fall back to yfinance
    for preferred in ["zerodha", "upstox", "angel", "groww", "yfinance"]:
        p = REGISTRY[preferred]
        if p.is_available():
            log.info("Auto-selected data provider: %s", p.name)
            return p

    raise RuntimeError("No data provider available. Install yfinance: pip install yfinance")
