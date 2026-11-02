import pytest
from services.risk_engine.calculator.risk_engine import RiskEngine, RiskInput


def _make_input(**overrides) -> RiskInput:
    defaults = dict(
        symbol="TCS",
        entry_price=4000.0,
        atr=40.0,
        signal_direction="BUY",
        portfolio_value=1_000_000.0,
        existing_exposure_pct=0.0,
        sector_exposure_pct=0.0,
    )
    return RiskInput(**{**defaults, **overrides})


class TestRiskEngine:

    def setup_method(self):
        self.engine = RiskEngine()

    def test_stoploss_below_entry_for_buy(self):
        out = self.engine.calculate(_make_input())
        assert out.stoploss < out.entry_price

    def test_target_above_entry_for_buy(self):
        out = self.engine.calculate(_make_input())
        assert out.target > out.entry_price

    def test_stoploss_above_entry_for_sell(self):
        out = self.engine.calculate(_make_input(signal_direction="SELL"))
        assert out.stoploss > out.entry_price

    def test_rr_ratio_above_minimum(self):
        out = self.engine.calculate(_make_input())
        assert out.risk_reward_ratio >= 1.5

    def test_position_size_respects_1pct_risk(self):
        out = self.engine.calculate(_make_input())
        risk_per_share = abs(out.entry_price - out.stoploss)
        max_loss = out.position_size_units * risk_per_share
        assert max_loss <= 1_000_000.0 * 0.011  # 1% + tiny floating point buffer

    def test_sector_overexposure_rejected(self):
        out = self.engine.calculate(_make_input(sector_exposure_pct=0.24, max_sector_pct=0.25))
        # position would push sector over 25%
        assert not out.is_valid or out.rejection_reason != ""

    def test_zero_atr_does_not_crash(self):
        out = self.engine.calculate(_make_input(atr=0.0))
        assert out is not None

    def test_output_prices_are_rounded(self):
        out = self.engine.calculate(_make_input())
        assert out.stoploss == round(out.stoploss, 2)
        assert out.target == round(out.target, 2)
