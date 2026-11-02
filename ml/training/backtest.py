"""
Backtester — simulates AlphaTrader signals on historical data.
Includes realistic transaction costs and slippage.

Usage:
    python ml/training/backtest.py --symbol RELIANCE TCS --days 730
    python ml/training/backtest.py --symbol all --days 365
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
from types import ModuleType

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if "numba" not in sys.modules:
    try:
        import numba  # noqa
    except ImportError:
        _s = ModuleType("numba")
        _s.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        sys.modules["numba"] = _s

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("backtest")

MIN_WARMUP = 250

# ── Realistic NSE transaction costs ──────────────────────────
# Brokerage (Zerodha flat): Rs.20 per order = Rs.40 round-trip
# STT (delivery): 0.1% on buy+sell = 0.2% total
# Exchange + SEBI + stamp: ~0.007%
# Estimated slippage: 0.05% each side for large-cap, 0.15% for small-cap
COST_PCT_LARGECAP  = 0.0030   # 0.30% round-trip (brokerage + STT + charges + slippage)
COST_PCT_SMALLCAP  = 0.0060   # 0.60% round-trip (higher slippage for small-caps)

LARGECAP_SYMBOLS = {
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC", "LT", "WIPRO", "HCLTECH",
    "AXISBANK", "ASIANPAINT", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "NESTLEIND", "NTPC", "BAJAJFINSV", "TECHM",
    "ONGC", "COALINDIA", "DIVISLAB", "DRREDDY", "CIPLA", "ADANIENT",
}


def get_cost_pct(symbol: str) -> float:
    return COST_PCT_LARGECAP if symbol.upper() in LARGECAP_SYMBOLS else COST_PCT_SMALLCAP


def backtest_symbol(symbol: str, df_full: pd.DataFrame, holding_days: int = 3) -> list[dict]:
    from predict import (
        compute_features, candlestick_score, technical_score,
        volume_score, ensemble_confidence, classify_signal,
        compute_risk, _chart_engine,
    )
    from ml.training.model_loader import ml_predict_proba

    cost_pct = get_cost_pct(symbol)
    trades   = []
    n        = len(df_full)

    for i in range(MIN_WARMUP, n - holding_days - 1):
        window = df_full.iloc[:i + 1].copy()
        try:
            feat_df = compute_features(window)
            if len(feat_df) < 5:
                continue

            row = feat_df.iloc[-1]

            ml_proba = ml_predict_proba(row.to_dict())
            if ml_proba:
                buy_p = ml_proba.get("BUY", 0.33)
                sel_p = ml_proba.get("SELL", 0.33)
                tech  = float(np.clip(buy_p - sel_p + 0.5, 0.0, 1.0))
            else:
                tech = technical_score(row)

            vol    = volume_score(row)
            candle = candlestick_score(feat_df)
            chart_score, _ = _chart_engine.detect_all(feat_df)
            conf   = ensemble_confidence(tech, vol, candle, chart=chart_score)
            signal, _ = classify_signal(conf, float(row["adx"]))

            if signal not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                continue

            # Entry: next-day open + half slippage
            raw_entry   = float(df_full["open"].iloc[i + 1])
            is_long     = "BUY" in signal
            slippage    = raw_entry * (cost_pct / 2)
            entry_price = raw_entry + slippage if is_long else raw_entry - slippage

            atr  = float(row["atr"])
            risk = compute_risk(entry_price, atr, signal)

            target   = risk["target"]
            stoploss = risk["sl"]
            outcome  = "EXPIRED"
            exit_price = float(df_full["close"].iloc[i + holding_days])

            for j in range(1, holding_days + 1):
                bar_high = float(df_full["high"].iloc[i + j])
                bar_low  = float(df_full["low"].iloc[i + j])
                if is_long:
                    if bar_high >= target:
                        outcome = "TARGET";   exit_price = target;   break
                    if bar_low  <= stoploss:
                        outcome = "STOPLOSS"; exit_price = stoploss; break
                else:
                    if bar_low  <= target:
                        outcome = "TARGET";   exit_price = target;   break
                    if bar_high >= stoploss:
                        outcome = "STOPLOSS"; exit_price = stoploss; break

            # Exit: apply half slippage on the way out
            exit_slippage = exit_price * (cost_pct / 2)
            exit_price    = exit_price - exit_slippage if is_long else exit_price + exit_slippage

            # P&L after ALL costs
            gross_pnl = (exit_price - entry_price) / entry_price * 100
            if not is_long:
                gross_pnl = -gross_pnl
            net_pnl = gross_pnl - cost_pct * 100   # subtract full round-trip cost

            trades.append({
                "symbol":     symbol,
                "date":       str(df_full.index[i + 1])[:10],
                "signal":     signal,
                "confidence": round(conf * 100, 1),
                "entry":      round(entry_price, 2),
                "target":     round(target, 2),
                "stoploss":   round(stoploss, 2),
                "exit":       round(exit_price, 2),
                "outcome":    outcome,
                "gross_pnl":  round(gross_pnl, 2),
                "cost_pct":   round(cost_pct * 100, 3),
                "net_pnl":    round(net_pnl, 2),
                "win":        net_pnl > 0,
            })

        except Exception as exc:
            log.debug("Signal error at bar %d for %s: %s", i, symbol, exc)

    return trades


def print_report(trades: list[dict]) -> None:
    if not trades:
        print("\n  No trades generated.\n")
        return

    df       = pd.DataFrame(trades)
    total    = len(df)
    wins     = df["win"].sum()
    win_rate = wins / total * 100

    avg_gross = df["gross_pnl"].mean()
    avg_net   = df["net_pnl"].mean()
    avg_cost  = df["cost_pct"].mean()
    best      = df["net_pnl"].max()
    worst     = df["net_pnl"].min()

    cumulative  = (1 + df["net_pnl"] / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = ((cumulative - rolling_max) / rolling_max * 100).min()
    returns     = df["net_pnl"].values
    sharpe      = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252 / 3)

    by_signal = df.groupby("signal").agg(
        count=("win", "count"),
        wins=("win", "sum"),
        avg_gross=("gross_pnl", "mean"),
        avg_net=("net_pnl", "mean"),
    )

    w = 65
    print("\n" + "=" * w)
    print("  ALPHATRADER BACKTEST RESULTS  (after transaction costs)")
    print("=" * w)
    print(f"  Total Trades     : {total}")
    print(f"  Win Rate         : {win_rate:.1f}%  ({int(wins)} wins / {total-int(wins)} losses)")
    print(f"  Avg Gross Return : {avg_gross:+.2f}%")
    print(f"  Avg Cost/Trade   : {avg_cost:.3f}%")
    print(f"  Avg Net Return   : {avg_net:+.2f}%  <-- what you actually make")
    print(f"  Best Trade (net) : {best:+.2f}%")
    print(f"  Worst Trade (net): {worst:+.2f}%")
    print(f"  Max Drawdown     : {drawdown:.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.2f}")
    print("-" * w)
    print(f"  {'Signal':<16} {'Trades':>6} {'Wins':>5} {'Win%':>6} {'Gross':>8} {'Net':>8}")
    print(f"  {'-'*55}")
    for sig, row in by_signal.iterrows():
        wr = row["wins"] / row["count"] * 100 if row["count"] > 0 else 0
        print(f"  {sig:<16} {int(row['count']):>6} {int(row['wins']):>5} "
              f"{wr:>5.1f}% {row['avg_gross']:>+7.2f}% {row['avg_net']:>+7.2f}%")
    print("=" * w)
    if avg_net < 0:
        print(f"\n  [!] Strategy loses money after costs. Consider raising confidence threshold.")
    elif avg_net < 0.3:
        print(f"\n  [~] Marginal edge after costs. Raise min confidence to 70%+ before live trading.")
    else:
        print(f"\n  [+] Strategy shows positive edge after costs.")
    print(f"\n  [!] Survivorship bias note: backtesting on current index members only.")
    print(f"      Stocks added to index recently will show inflated historical performance.\n")

    out = Path("backtest_results.csv")
    df.to_csv(out, index=False)
    print(f"  Full results saved to: {out.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="AlphaTrader Backtester")
    parser.add_argument("--symbol",  nargs="+", default=["RELIANCE"])
    parser.add_argument("--days",    type=int,  default=730)
    parser.add_argument("--holding", type=int,  default=3)
    args = parser.parse_args()

    from predict import resolve_symbols
    from services.ingestion.adapters.yfinance_adapter import fetch_ohlcv

    symbols, label = resolve_symbols(args.symbol)
    log.info("Backtesting %s | %dd history | %dd hold", label, args.days, args.holding)

    all_trades = []
    for sym in symbols:
        try:
            log.info("Backtesting %s ...", sym)
            df     = fetch_ohlcv(sym, interval="day", days=args.days)
            trades = backtest_symbol(sym, df, holding_days=args.holding)
            log.info("  %s: %d trades", sym, len(trades))
            all_trades.extend(trades)
        except Exception as exc:
            log.warning("Skipping %s: %s", sym, exc)

    print_report(all_trades)


if __name__ == "__main__":
    main()
