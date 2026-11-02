"""
Risk management engine.
Calculates position size, target, stoploss, and risk-reward ratio
using ATR-based dynamic levels and fixed fractional position sizing.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

MAX_RISK_PER_TRADE_PCT = 0.01  # 1% of portfolio
ATR_STOPLOSS_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 3.0
MIN_RISK_REWARD = 1.5


@dataclass
class RiskInput:
    symbol: str
    entry_price: float
    atr: float
    signal_direction: str       # "BUY" or "SELL"
    portfolio_value: float
    existing_exposure_pct: float = 0.0
    sector_exposure_pct: float = 0.0
    max_sector_pct: float = 0.25
    max_portfolio_exposure_pct: float = 0.80


@dataclass
class RiskOutput:
    entry_price: float
    stoploss: float
    target: float
    risk_reward_ratio: float
    position_size_pct: float
    position_size_units: int
    max_drawdown_estimate: float
    is_valid: bool
    rejection_reason: str = ""


class RiskEngine:

    def calculate(self, inp: RiskInput) -> RiskOutput:
        is_long = inp.signal_direction in ("BUY", "STRONG_BUY")

        stoploss = (
            inp.entry_price - (inp.atr * ATR_STOPLOSS_MULTIPLIER)
            if is_long
            else inp.entry_price + (inp.atr * ATR_STOPLOSS_MULTIPLIER)
        )
        target = (
            inp.entry_price + (inp.atr * ATR_TARGET_MULTIPLIER)
            if is_long
            else inp.entry_price - (inp.atr * ATR_TARGET_MULTIPLIER)
        )

        risk_per_share = abs(inp.entry_price - stoploss)
        reward_per_share = abs(target - inp.entry_price)
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

        max_risk_amount = inp.portfolio_value * MAX_RISK_PER_TRADE_PCT
        position_size_units = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = position_size_units * inp.entry_price
        position_size_pct = position_value / inp.portfolio_value

        is_valid, reason = self._validate(inp, rr_ratio, position_size_pct)

        return RiskOutput(
            entry_price=inp.entry_price,
            stoploss=round(stoploss, 2),
            target=round(target, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            position_size_pct=round(position_size_pct, 4),
            position_size_units=position_size_units,
            max_drawdown_estimate=round(position_size_pct * (risk_per_share / inp.entry_price), 4),
            is_valid=is_valid,
            rejection_reason=reason,
        )

    @staticmethod
    def _validate(inp: RiskInput, rr: float, pos_pct: float) -> tuple[bool, str]:
        if rr < MIN_RISK_REWARD:
            return False, f"R:R {rr:.2f} below minimum {MIN_RISK_REWARD}"
        if inp.sector_exposure_pct + pos_pct > inp.max_sector_pct:
            return False, f"Sector exposure would exceed {inp.max_sector_pct*100:.0f}%"
        if inp.existing_exposure_pct + pos_pct > inp.max_portfolio_exposure_pct:
            return False, f"Portfolio exposure would exceed {inp.max_portfolio_exposure_pct*100:.0f}%"
        return True, ""
