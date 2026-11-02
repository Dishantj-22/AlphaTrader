"""
Lightweight local news sentiment cache for predict.py.
Fetches recent NSE announcements and financial headlines,
scores them with a simple keyword-based model (no GPU needed),
and caches results to a local JSON file.

For full FinBERT-powered sentiment, the microservice in
services/news_sentiment/ handles that at scale.
This module gives predict.py a fast, offline-capable fallback.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CACHE_FILE    = Path("tracker/sentiment_cache.json")
CACHE_TTL_HRS = 4    # refresh every 4 hours

# ── Keyword-based sentiment scorer ───────────────────────────
# Positive keywords found in financial news
BULLISH_KEYWORDS = [
    "wins contract", "wins order", "new order", "order win", "partnership",
    "joint venture", "acquisition", "buyback", "dividend", "bonus",
    "strong results", "beat estimates", "profit up", "revenue up",
    "expansion", "capacity addition", "new launch", "fda approval",
    "upgrade", "outperform", "strong buy", "target raised",
    "record revenue", "record profit", "market share gain",
    "debt free", "debt reduction", "promoter buying",
]

BEARISH_KEYWORDS = [
    "profit warning", "revenue miss", "below estimates", "loss",
    "downgrade", "underperform", "reduce", "target cut",
    "investigation", "sebi notice", "fraud", "penalty",
    "debt increase", "promoter selling", "stake sale",
    "plant shutdown", "strike", "recall", "lawsuit",
    "resignation", "management change", "margin pressure",
    "default", "npa", "write-off",
]


def _keyword_sentiment(text: str) -> float:
    """Returns sentiment score -1 to +1 based on keyword matching."""
    text_lower = text.lower()
    bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _load_cache() -> dict:
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        log.debug("Could not save sentiment cache: %s", exc)


def _is_stale(entry: dict) -> bool:
    try:
        ts = datetime.fromisoformat(entry.get("fetched_at", "2000-01-01"))
        return datetime.now() - ts > timedelta(hours=CACHE_TTL_HRS)
    except Exception:
        return True


def fetch_symbol_sentiment(symbol: str) -> float:
    """
    Returns sentiment score for a symbol (-1 to +1).
    Uses cached value if fresh, otherwise fetches NSE announcements.
    """
    cache = _load_cache()
    entry = cache.get(symbol.upper())

    if entry and not _is_stale(entry):
        return float(entry["score"])

    score = _fetch_fresh_sentiment(symbol)
    cache[symbol.upper()] = {
        "score":      round(score, 3),
        "fetched_at": datetime.now().isoformat(),
    }
    _save_cache(cache)
    return score


def _fetch_fresh_sentiment(symbol: str) -> float:
    """Fetch NSE corporate announcements and score with keywords."""
    try:
        import requests
        import time

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Referer": "https://www.nseindia.com/",
        })
        session.get("https://www.nseindia.com", timeout=8)
        time.sleep(1.0)

        url  = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol.upper()}"
        resp = session.get(url, timeout=10)

        if resp.status_code != 200:
            return 0.0

        announcements = resp.json()
        if not announcements:
            return 0.0

        # Score latest 5 announcements, weight recent ones more
        scores = []
        for i, ann in enumerate(announcements[:5]):
            text  = ann.get("desc", "") + " " + ann.get("attchmntText", "")
            score = _keyword_sentiment(text)
            weight = 1.0 / (i + 1)   # more recent = higher weight
            scores.append((score, weight))

        if not scores:
            return 0.0

        total_w = sum(w for _, w in scores)
        return sum(s * w for s, w in scores) / total_w

    except Exception as exc:
        log.debug("Sentiment fetch failed for %s: %s", symbol, exc)
        return 0.0


def get_sentiment_for_predict(symbol: str,
                               fii_dii_sentiment: float = 0.0,
                               options_sentiment: float = 0.0) -> tuple[float, list[str]]:
    """
    Combined sentiment score for predict.py.
    Blends:
      - NSE announcements (keyword-based): 40%
      - FII/DII flow: 35%
      - Options PCR: 25%
    Returns (score 0-1, list of sources)
    """
    news_raw  = fetch_symbol_sentiment(symbol)
    news_norm = (news_raw + 1) / 2.0   # -1..+1 → 0..1

    # FII/DII and options are already 0-1 scaled by caller
    fii_norm  = (fii_dii_sentiment + 1) / 2.0
    opt_norm  = (options_sentiment + 1) / 2.0

    combined = news_norm * 0.40 + fii_norm * 0.35 + opt_norm * 0.25

    sources = []
    if abs(news_raw) > 0.1:
        direction = "positive" if news_raw > 0 else "negative"
        sources.append(f"News: {direction} ({news_raw:+.2f})")

    return round(combined, 3), sources
