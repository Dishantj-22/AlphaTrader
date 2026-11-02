"""
AlphaTrader — Signal Outcome Tracker

Run this every evening after market close (3:30 PM IST).
It fetches today's closing prices, checks all open signals,
marks them as TARGET HIT / STOPLOSS HIT / OPEN / EXPIRED,
and updates the dynamic weight learning database.

Usage:
    python track.py                     # check all open signals
    python track.py --add               # manually add a signal
    python track.py --journal           # show performance journal
    python track.py --stats             # show full statistics
    python track.py --export            # export to CSV
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from types import ModuleType

# Numba stub for Python 3.14
if "numba" not in sys.modules:
    try:
        import numba  # noqa
    except ImportError:
        _s = ModuleType("numba")
        _s.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        sys.modules["numba"] = _s

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("tracker")

# ── Storage ───────────────────────────────────────────────────
SIGNALS_FILE = Path("tracker/signals.json")
SIGNALS_FILE.parent.mkdir(exist_ok=True)


# ── Data Layer ────────────────────────────────────────────────

def load_signals() -> list[dict]:
    try:
        if SIGNALS_FILE.exists():
            with open(SIGNALS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_signals(signals: list[dict]) -> None:
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2)


def get_current_price(symbol: str) -> float | None:
    try:
        import yfinance as yf
        df = yf.download(symbol + ".NS", period="2d", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return float(df["close"].dropna().iloc[-1])
    except Exception:
        return None


import pandas as pd


# ── Core Logic ────────────────────────────────────────────────

def add_signal_interactive() -> None:
    """Manually add a signal to track."""
    print("\n  Add Signal to Tracker")
    print("  " + "-" * 40)
    symbol    = input("  Symbol         : ").strip().upper()
    signal    = input("  Signal (BUY/SELL): ").strip().upper()
    entry     = float(input("  Entry price    : Rs.").strip())
    target    = float(input("  Target price   : Rs.").strip())
    stoploss  = float(input("  Stop Loss      : Rs.").strip())
    holding   = int(input("  Holding days   : ").strip())
    conf      = float(input("  Confidence %   : ").strip())

    record = {
        "id":           f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}",
        "symbol":       symbol,
        "signal":       signal,
        "entry":        entry,
        "target":       target,
        "stoploss":     stoploss,
        "holding_days": holding,
        "confidence":   conf,
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "expiry_date":  (datetime.now() + timedelta(days=holding)).strftime("%Y-%m-%d"),
        "status":       "OPEN",
        "exit_price":   None,
        "outcome":      None,
        "pnl_pct":      None,
        "closed_date":  None,
    }

    signals = load_signals()
    signals.append(record)
    save_signals(signals)

    print(f"\n  [+] Signal added: {symbol} {signal} @ Rs.{entry}")
    print(f"      Target: Rs.{target}  |  SL: Rs.{stoploss}  |  Expiry: {record['expiry_date']}")


def check_open_signals() -> None:
    """Fetch current prices and update open signal statuses."""
    signals  = load_signals()
    open_sig = [s for s in signals if s["status"] == "OPEN"]

    if not open_sig:
        print("\n  No open signals to check.\n")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    updated = 0

    print(f"\n  Checking {len(open_sig)} open signal(s)...\n")

    for sig in open_sig:
        symbol  = sig["symbol"]
        price   = get_current_price(symbol)

        if price is None:
            print(f"  [{symbol}] Could not fetch price — skipping")
            continue

        is_long   = "BUY" in sig["signal"]
        target    = sig["target"]
        stoploss  = sig["stoploss"]
        entry     = sig["entry"]
        expiry    = sig["expiry_date"]
        outcome   = None
        exit_price = price

        # Check target
        if is_long and price >= target:
            outcome    = "TARGET"
            exit_price = target
        elif not is_long and price <= target:
            outcome    = "TARGET"
            exit_price = target
        # Check stoploss
        elif is_long and price <= stoploss:
            outcome    = "STOPLOSS"
            exit_price = stoploss
        elif not is_long and price >= stoploss:
            outcome    = "STOPLOSS"
            exit_price = stoploss
        # Check expiry
        elif today >= expiry:
            outcome    = "EXPIRED"
            exit_price = price

        if outcome:
            pnl = (exit_price - entry) / entry * 100
            if not is_long:
                pnl = -pnl

            sig["status"]      = "CLOSED"
            sig["outcome"]     = outcome
            sig["exit_price"]  = round(exit_price, 2)
            sig["pnl_pct"]     = round(pnl, 2)
            sig["closed_date"] = today

            icon = "[+]" if pnl > 0 else "[-]"
            print(f"  {icon} {symbol:<12} {sig['signal']:<8} "
                  f"Entry Rs.{entry:,.0f} → Exit Rs.{exit_price:,.0f} "
                  f"| {outcome:<8} | {pnl:+.2f}%")

            # Update dynamic weights
            _update_dynamic_weights(sig)
            updated += 1
        else:
            pnl_live = (price - entry) / entry * 100
            if not is_long:
                pnl_live = -pnl_live
            icon = "[~]" if pnl_live > 0 else "[~]"
            print(f"  {icon} {symbol:<12} {sig['signal']:<8} "
                  f"Entry Rs.{entry:,.0f} → Now Rs.{price:,.0f} "
                  f"| OPEN ({pnl_live:+.1f}%) | Expiry {expiry}")

    if updated:
        save_signals(signals)
        print(f"\n  {updated} signal(s) closed and saved.")
    else:
        print("\n  No signals closed today.")


def _update_dynamic_weights(sig: dict) -> None:
    """Push closed signal outcome to the dynamic weight tracker."""
    try:
        from services.signal_engine.dynamic_weights import record_outcome
        record_outcome(
            symbol=sig["symbol"],
            date=sig["date"],
            outcome=sig["outcome"],
            pnl_pct=sig.get("pnl_pct", 0.0),
        )
    except Exception as exc:
        log.debug("Dynamic weight update failed: %s", exc)


# ── Journal & Stats ───────────────────────────────────────────

def show_journal(days: int = 30) -> None:
    """Print recent trade journal."""
    signals  = load_signals()
    cutoff   = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent   = [s for s in signals
                if s.get("closed_date", "9999") >= cutoff and s["status"] == "CLOSED"]

    if not recent:
        print(f"\n  No closed trades in last {days} days.\n")
        return

    recent.sort(key=lambda x: x.get("closed_date", ""))

    w = 72
    print("\n" + "=" * w)
    print(f"  TRADE JOURNAL  (last {days} days — {len(recent)} trades)")
    print("=" * w)
    print(f"  {'DATE':<12} {'SYMBOL':<12} {'SIGNAL':<8} {'CONF':>5} "
          f"{'ENTRY':>8} {'EXIT':>8} {'OUTCOME':<10} {'PNL':>7}")
    print("  " + "-" * 68)

    for s in recent:
        icon = "[+]" if (s.get("pnl_pct") or 0) > 0 else "[-]"
        print(f"  {icon} {s.get('closed_date','?'):<10} {s['symbol']:<12} "
              f"{s['signal']:<8} {s['confidence']:>4.0f}% "
              f"Rs.{s['entry']:>7,.0f} Rs.{s.get('exit_price',0):>7,.0f} "
              f"{s.get('outcome','?'):<10} {s.get('pnl_pct',0):>+6.1f}%")

    print("=" * w)


def show_stats() -> None:
    """Show full performance statistics."""
    signals = load_signals()
    closed  = [s for s in signals if s["status"] == "CLOSED" and s.get("pnl_pct") is not None]

    if not closed:
        print("\n  No closed trades yet. Add signals and run --check to build history.\n")
        return

    total    = len(closed)
    wins     = [s for s in closed if s.get("pnl_pct", 0) > 0]
    losses   = [s for s in closed if s.get("pnl_pct", 0) <= 0]
    targets  = [s for s in closed if s.get("outcome") == "TARGET"]
    stops    = [s for s in closed if s.get("outcome") == "STOPLOSS"]
    expired  = [s for s in closed if s.get("outcome") == "EXPIRED"]

    win_rate  = len(wins) / total * 100
    avg_win   = np.mean([s["pnl_pct"] for s in wins])   if wins   else 0
    avg_loss  = np.mean([s["pnl_pct"] for s in losses]) if losses else 0
    avg_pnl   = np.mean([s["pnl_pct"] for s in closed])
    total_pnl = sum(s["pnl_pct"] for s in closed)

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    loss_rate  = (100 - win_rate) / 100
    expectancy = (win_rate / 100 * avg_win) + (loss_rate * avg_loss)

    # Sharpe (approximate)
    pnls   = np.array([s["pnl_pct"] for s in closed])
    sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-9)) * np.sqrt(252 / 3)

    # Best / worst
    best  = max(closed, key=lambda x: x["pnl_pct"])
    worst = min(closed, key=lambda x: x["pnl_pct"])

    # By signal type
    by_signal: dict[str, list] = {}
    for s in closed:
        sig = s["signal"]
        by_signal.setdefault(sig, []).append(s["pnl_pct"])

    # Streak
    streak, best_streak, cur = 0, 0, 0
    for s in sorted(closed, key=lambda x: x.get("closed_date", "")):
        if s["pnl_pct"] > 0:
            cur += 1
            best_streak = max(best_streak, cur)
        else:
            cur = 0

    w = 65
    print("\n" + "=" * w)
    print("  ALPHATRADER PERFORMANCE STATISTICS")
    print("=" * w)
    print(f"  Total Trades     : {total}")
    print(f"  Win Rate         : {win_rate:.1f}%  ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Target Hit       : {len(targets)}  |  SL Hit: {len(stops)}  |  Expired: {len(expired)}")
    print(f"  Avg Win          : +{avg_win:.2f}%")
    print(f"  Avg Loss         : {avg_loss:.2f}%")
    print(f"  Avg Trade (net)  : {avg_pnl:+.2f}%")
    print(f"  Total Return     : {total_pnl:+.2f}%")
    print(f"  Expectancy/Trade : {expectancy:+.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.2f}")
    print(f"  Best Trade       : {best['symbol']} {best['pnl_pct']:+.1f}% ({best.get('closed_date','')})")
    print(f"  Worst Trade      : {worst['symbol']} {worst['pnl_pct']:+.1f}% ({worst.get('closed_date','')})")
    print(f"  Best Win Streak  : {best_streak}")
    print("-" * w)
    print(f"  {'Signal':<14} {'Trades':>6} {'Win%':>6} {'Avg PnL':>9}")
    print(f"  {'-' * 40}")
    for sig, pnls in sorted(by_signal.items()):
        sig_wins = sum(1 for p in pnls if p > 0)
        print(f"  {sig:<14} {len(pnls):>6} {sig_wins/len(pnls)*100:>5.0f}% {np.mean(pnls):>+8.2f}%")
    print("=" * w)

    # Verdict
    print()
    if expectancy > 0.5 and win_rate > 55:
        print("  [+] System has a positive edge. Keep trading and tracking.")
    elif expectancy > 0 and win_rate > 45:
        print("  [~] Marginal edge. Consider raising confidence threshold to 70%+.")
    else:
        print("  [-] No clear edge yet. Run more trades or retrain the ML model.")

    if total < 30:
        print(f"  [!] Only {total} trades — need 30+ for statistically meaningful results.")
    print()


def export_csv() -> None:
    """Export all signals to CSV."""
    signals = load_signals()
    if not signals:
        print("\n  No signals to export.\n")
        return
    df = pd.DataFrame(signals)
    path = Path("tracker/signals_export.csv")
    df.to_csv(path, index=False)
    print(f"\n  Exported {len(signals)} signals to {path.resolve()}\n")


def show_open() -> None:
    """Show currently open signals."""
    signals  = load_signals()
    open_sig = [s for s in signals if s["status"] == "OPEN"]
    if not open_sig:
        print("\n  No open signals.\n")
        return

    w = 70
    print("\n" + "=" * w)
    print(f"  OPEN SIGNALS  ({len(open_sig)} active)")
    print("=" * w)
    print(f"  {'DATE':<12} {'SYMBOL':<12} {'SIGNAL':<8} "
          f"{'ENTRY':>9} {'TARGET':>9} {'SL':>9} {'EXPIRY':<12}")
    print("  " + "-" * 66)
    for s in open_sig:
        print(f"  {s['date']:<12} {s['symbol']:<12} {s['signal']:<8} "
              f"Rs.{s['entry']:>7,.0f} Rs.{s['target']:>7,.0f} "
              f"Rs.{s['stoploss']:>7,.0f} {s['expiry_date']}")
    print("=" * w + "\n")


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaTrader Signal Outcome Tracker",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Commands:
  python track.py               Check open signals against current prices
  python track.py --add         Manually add a signal to track
  python track.py --open        Show all currently open signals
  python track.py --journal     Show trade journal (last 30 days)
  python track.py --journal 90  Show trade journal (last 90 days)
  python track.py --stats       Show full performance statistics
  python track.py --export      Export all signals to CSV

Daily workflow:
  Morning  : python predict.py --top5 --style btst
             python track.py --add   (add the signals you act on)
  Evening  : python track.py         (check outcomes)
  Weekly   : python track.py --stats (review performance)
        """
    )
    parser.add_argument("--add",     action="store_true", help="Add a new signal")
    parser.add_argument("--open",    action="store_true", help="Show open signals")
    parser.add_argument("--journal", nargs="?", const=30, type=int,
                        metavar="DAYS", help="Show journal (default 30 days)")
    parser.add_argument("--stats",   action="store_true", help="Performance statistics")
    parser.add_argument("--export",  action="store_true", help="Export to CSV")
    args = parser.parse_args()

    if args.add:
        add_signal_interactive()
    elif args.open:
        show_open()
    elif args.journal is not None:
        show_journal(days=args.journal)
    elif args.stats:
        show_stats()
    elif args.export:
        export_csv()
    else:
        # Default: check open signals
        check_open_signals()
        # Also show open positions after checking
        show_open()


if __name__ == "__main__":
    main()
