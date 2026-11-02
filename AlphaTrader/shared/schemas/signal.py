from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TradingStyle(str, Enum):
    INTRADAY = "INTRADAY"
    BTST = "BTST"
    SWING = "SWING"


class ModelContribution(BaseModel):
    model_name: str
    signal: SignalType
    confidence: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)


class TradingSignal(BaseModel):
    signal_id: str
    symbol: str
    exchange: str
    generated_at: datetime
    trading_style: TradingStyle
    signal: SignalType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Ensemble confidence 0-1")
    entry_price: float
    target_price: float
    stoploss_price: float
    risk_reward_ratio: float
    holding_days: int
    position_size_pct: float = Field(..., description="% of capital to allocate")
    model_contributions: list[ModelContribution]
    technicals_score: float
    sentiment_score: float
    volume_score: float
    pattern_score: float
    reasoning: Optional[str] = None
    is_valid: bool = True

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stoploss_price)

    @property
    def reward_per_share(self) -> float:
        return abs(self.target_price - self.entry_price)


class SignalAlert(BaseModel):
    signal: TradingSignal
    alert_channel: list[str] = Field(default_factory=lambda: ["push", "websocket"])
    notified_at: Optional[datetime] = None
