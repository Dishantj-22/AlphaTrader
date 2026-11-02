"""
Earnings & Corporate Event Calendar filter.
Fetches upcoming results dates from NSE corporate actions API.
Prevents generating signals within N days of an earnings announcement
where gap risk is extreme and models have no predictive power.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

NSE_CORPORATE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={symbol}"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

EARNINGS_BLOCK_DAYS = 3    # block signals this many days before results
DIVIDEND_BLOCK_DAYS = 2    # block 2 days before ex-dividend (gap risk)


def _nse_session() -> requests.Session:
    import time
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1.0)
        session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return session


def get_upcoming_events(symbol: str) -> list[dict]:
    """
    Returns list of upcoming corporate events for the symbol.
    Each event: {"type": str, "date": datetime, "description": str}
    """
    try:
        session = _nse_session()
        url  = NSE_CORPORATE_ACTIONS_URL.format(symbol=symbol.upper())
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        events = []
        today  = datetime.now().date()

        for item in (data or []):
            ex_date_str = item.get("exDate", "") or item.get("recordDate", "")
            purpose     = item.get("purpose", "").upper()
            if not ex_date_str:
                continue
            try:
                ex_date = datetime.strptime(ex_date_str, "%d-%b-%Y").date()
            except ValueError:
                continue

            if ex_date < today or ex_date > today + timedelta(days=30):
                continue

            event_type = "OTHER"
            if any(k in purpose for k in ["RESULT", "QUARTERLY", "ANNUAL", "Q1", "Q2", "Q3", "Q4"]):
                event_type = "EARNINGS"
            elif any(k in purpose for k in ["DIVIDEND", "INTERIM DIVIDEND"]):
                event_type = "DIVIDEND"
            elif "BONUS" in purpose:
                event_type = "BONUS"
            elif "SPLIT" in purpose:
                event_type = "SPLIT"

            events.append({
                "type":        event_type,
                "date":        ex_date,
                "description": item.get("purpose", ""),
            })

        return events

    except Exception as exc:
        log.debug("Corporate action fetch failed for %s: %s", symbol, exc)
        return []


def is_safe_to_trade(symbol: str, check_date: Optional[datetime] = None) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    is_safe=False means there's an upcoming earnings/event — skip the signal.
    """
    today  = (check_date or datetime.now()).date()
    events = get_upcoming_events(symbol)

    for event in events:
        days_away = (event["date"] - today).days

        if event["type"] == "EARNINGS" and 0 <= days_away <= EARNINGS_BLOCK_DAYS:
            return False, (
                f"{symbol} earnings in {days_away}d ({event['date']}) — "
                f"gap risk too high, signal blocked"
            )
        if event["type"] == "DIVIDEND" and 0 <= days_away <= DIVIDEND_BLOCK_DAYS:
            return False, (
                f"{symbol} ex-dividend in {days_away}d ({event['date']}) — "
                f"price adjustment risk, signal blocked"
            )
        if event["type"] in ("BONUS", "SPLIT") and 0 <= days_away <= 1:
            return False, (
                f"{symbol} {event['type'].lower()} in {days_away}d — "
                f"price will adjust, signal blocked"
            )

    return True, ""
