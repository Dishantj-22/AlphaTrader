"""
Dynamic Ensemble Weight Manager.
Tracks recent signal performance and adjusts sub-model weights
based on which models were most accurate in the last 30 days.

Default weights (no history):
  technical     35%
  chart_pattern 20%
  candlestick   10%
  volume        15%
  sentiment     20%

After 30+ tracked signals: weights shift toward best-performing models.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

HISTORY_FILE = Path(__file__).parent / "ensemble_history.json"

DEFAULT_WEIGHTS = {
    "technical":     0.35,
    "chart_pattern": 0.20,
    "candlestick":   0.10,
    "volume":        0.15,
    "sentiment":     0.20,
}

MIN_WEIGHT  = 0.05   # no model drops below 5%
MAX_WEIGHT  = 0.60   # no model exceeds 60%
LOOKBACK_DAYS = 30
MIN_SAMPLES   = 20   # need at least 20 resolved trades before adapting


def load_history() -> list[dict]:
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_signal(signal_data: dict) -> None:
    """
    Call this when a signal is generated.
    signal_data: {symbol, date, signal, confidence, scores: {tech, chart, candle, vol, sentiment}}
    """
    history = load_history()
    history.append({**signal_data, "recorded_at": datetime.now().isoformat()})
    # Keep last 500 entries
    history = history[-500:]
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as exc:
        log.warning("Could not save signal history: %s", exc)


def record_outcome(symbol: str, date: str, outcome: str, pnl_pct: float) -> None:
    """
    Call this when you know the trade outcome (TARGET/STOPLOSS/EXPIRED).
    Allows the system to learn which models were right.
    """
    history = load_history()
    for entry in history:
        if entry.get("symbol") == symbol and entry.get("date") == date:
            entry["outcome"] = outcome
            entry["pnl_pct"] = pnl_pct
            entry["resolved_at"] = datetime.now().isoformat()
            break
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as exc:
        log.warning("Could not update outcome: %s", exc)


def get_dynamic_weights() -> dict[str, float]:
    """
    Returns weight dict adjusted by recent performance.
    Falls back to defaults if insufficient history.
    """
    history  = load_history()
    cutoff   = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    recent   = [
        h for h in history
        if h.get("outcome") and h.get("pnl_pct") is not None
        and datetime.fromisoformat(h.get("recorded_at", "2000-01-01")) > cutoff
    ]

    if len(recent) < MIN_SAMPLES:
        log.debug("Insufficient history (%d samples) — using default weights", len(recent))
        return DEFAULT_WEIGHTS.copy()

    # For each sub-model, compute correlation between its score and trade outcome
    model_keys = ["technical", "chart_pattern", "candlestick", "volume", "sentiment"]
    performance = {}

    for key in model_keys:
        scores   = []
        outcomes = []
        for entry in recent:
            score = entry.get("scores", {}).get(key)
            pnl   = entry.get("pnl_pct")
            if score is not None and pnl is not None:
                # For BUY signals: high score + positive PnL = good
                # Normalise score to 0-1 range
                scores.append(float(score) / 100)
                outcomes.append(1.0 if pnl > 0 else 0.0)

        if len(scores) < 10:
            performance[key] = 0.5   # neutral
        else:
            import numpy as np
            corr = float(np.corrcoef(scores, outcomes)[0, 1])
            performance[key] = max(0.0, (corr + 1) / 2)   # shift from [-1,1] to [0,1]

    # Convert performance to weights
    total_perf = sum(performance.values()) or 1.0
    raw_weights = {k: v / total_perf for k, v in performance.items()}

    # Clip to min/max
    clipped = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in raw_weights.items()}
    total   = sum(clipped.values())
    weights = {k: round(v / total, 4) for k, v in clipped.items()}

    log.info("Dynamic weights (from %d samples): %s", len(recent), weights)
    return weights
