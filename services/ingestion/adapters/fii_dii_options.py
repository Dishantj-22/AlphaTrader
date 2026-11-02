"""
FII/DII and Options chain fetcher.
Data sourced from NSE public endpoints — no API key required.

Provides:
  - FII/DII net buy/sell activity (cash market)
  - Nifty Options Put/Call ratio
  - Max Pain level for Nifty
"""

import logging
import requests
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

log = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
}

NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"


@dataclass
class FIIDIIData:
    date: str
    fii_net_buy: float
    dii_net_buy: float
    combined_net: float
    sentiment: float          # -1 to +1
    message: str


@dataclass
class OptionsData:
    pcr: float
    max_pain: float
    strong_support: float
    strong_resistance: float
    sentiment: float          # -1 to +1
    message: str


def _nse_session() -> requests.Session:
    import time
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    # NSE uses Akamai bot protection — needs two warm-up hits with deliberate delays
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1.5)
        session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=10)
        time.sleep(1.0)
    except Exception:
        pass
    return session


def fetch_fii_dii() -> Optional[FIIDIIData]:
    try:
        session = _nse_session()
        resp = session.get(NSE_FII_DII_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        latest  = data[0] if data else {}
        # NSE returns separate DII and FII/FPI rows — pick by category
        fii_row = next((d for d in data if "FII" in d.get("category", "")), data[0] if data else {})
        dii_row = next((d for d in data if d.get("category", "") == "DII"), {})

        fii_net  = float(fii_row.get("netValue", 0) or 0)
        dii_net  = float(dii_row.get("netValue", 0) or 0)
        combined = fii_net + dii_net
        date_str = fii_row.get("date", datetime.now().strftime("%d-%b-%Y"))

        cap       = 5000.0
        sentiment = max(-1.0, min(1.0, combined / cap))

        if combined > 500:
            msg = f"FII+DII net buying Rs.{combined:.0f} Cr — institutional bullish"
        elif combined < -500:
            msg = f"FII+DII net selling Rs.{abs(combined):.0f} Cr — institutional bearish"
        else:
            msg = f"FII/FPI {fii_net:+.0f} Cr | DII {dii_net:+.0f} Cr — mixed"

        return FIIDIIData(
            date=date_str,
            fii_net_buy=fii_net,
            dii_net_buy=dii_net,
            combined_net=combined,
            sentiment=round(sentiment, 3),
            message=msg,
        )
    except Exception as exc:
        log.warning("FII/DII fetch failed: %s", exc)
        return None


def fetch_options_data() -> Optional[OptionsData]:
    """
    Fetch Nifty Options PCR and max pain.
    Primary: NSE option-chain API (works when NSE session is active).
    Fallback: Derive a synthetic PCR from India VIX level.
    """
    try:
        session = _nse_session()
        resp    = session.get(NSE_OPTIONS_URL, timeout=12)

        # NSE sometimes returns 404 when the endpoint is rate-limited or path changed
        if resp.status_code != 200:
            return _synthetic_options_from_vix()

        data    = resp.json()
        records = data.get("records", {}).get("data", [])
        if not records:
            return _synthetic_options_from_vix()

        ce_oi: dict[float, float] = {}
        pe_oi: dict[float, float] = {}
        total_ce_oi = 0.0
        total_pe_oi = 0.0

        for rec in records:
            strike = float(rec.get("strikePrice", 0))
            if "CE" in rec:
                oi = float(rec["CE"].get("openInterest", 0) or 0)
                ce_oi[strike] = oi
                total_ce_oi += oi
            if "PE" in rec:
                oi = float(rec["PE"].get("openInterest", 0) or 0)
                pe_oi[strike] = oi
                total_pe_oi += oi

        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

        strikes    = sorted(set(list(ce_oi.keys()) + list(pe_oi.keys())))
        pain_values = {}
        for s in strikes:
            ce_loss = sum(max(0, s - k) * v for k, v in ce_oi.items())
            pe_loss = sum(max(0, k - s) * v for k, v in pe_oi.items())
            pain_values[s] = ce_loss + pe_loss
        max_pain   = min(pain_values, key=pain_values.get) if pain_values else 0
        support    = max(pe_oi, key=pe_oi.get) if pe_oi else 0
        resistance = max(ce_oi, key=ce_oi.get) if ce_oi else 0

        if pcr > 1.2:
            sentiment = min(1.0, (pcr - 1.0) / 0.5)
            msg = f"PCR {pcr:.2f} — Bullish. Put support at Rs.{support:.0f}, max pain Rs.{max_pain:.0f}"
        elif pcr < 0.8:
            sentiment = max(-1.0, -(1.0 - pcr) / 0.5)
            msg = f"PCR {pcr:.2f} — Bearish. Call resistance at Rs.{resistance:.0f}, max pain Rs.{max_pain:.0f}"
        else:
            sentiment = 0.0
            msg = f"PCR {pcr:.2f} — Neutral. Max pain Rs.{max_pain:.0f}, support Rs.{support:.0f}"

        return OptionsData(
            pcr=round(pcr, 2),
            max_pain=max_pain,
            strong_support=support,
            strong_resistance=resistance,
            sentiment=round(sentiment, 3),
            message=msg,
        )
    except Exception as exc:
        log.warning("Options data fetch failed: %s — using VIX-based fallback", exc)
        return _synthetic_options_from_vix()


def _synthetic_options_from_vix() -> Optional[OptionsData]:
    """
    When the NSE options endpoint is unavailable, derive a synthetic PCR
    from India VIX. Low VIX = complacency = more calls than puts = PCR < 1.
    High VIX = fear = hedging with puts = PCR > 1.
    This is an approximation, not real options data.
    """
    try:
        import yfinance as yf
        import pandas as pd
        vix_df = yf.download("^INDIAVIX", period="3d", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = [c[0].lower() for c in vix_df.columns]
        else:
            vix_df.columns = [c.lower() for c in vix_df.columns]

        vix = float(vix_df["close"].dropna().iloc[-1]) if not vix_df.empty else 15.0

        # Empirical mapping: VIX 12 → PCR ~0.85, VIX 18 → PCR ~1.15, VIX 25 → PCR ~1.40
        pcr       = 0.60 + (vix / 25.0) * 0.80
        pcr       = round(min(1.8, max(0.5, pcr)), 2)
        sentiment = max(-1.0, min(1.0, (pcr - 1.0) / 0.4))
        msg       = f"PCR ~{pcr:.2f} (VIX-derived, NSE endpoint unavailable) — {'Bullish' if pcr > 1.1 else 'Bearish' if pcr < 0.9 else 'Neutral'}"

        return OptionsData(
            pcr=pcr, max_pain=0, strong_support=0, strong_resistance=0,
            sentiment=round(sentiment, 3), message=msg,
        )
    except Exception:
        return None
