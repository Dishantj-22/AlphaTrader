from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class OHLCVCandle(BaseModel):
    symbol: str
    exchange: Exchange
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None
    vwap: Optional[float] = None
    delivery_qty: Optional[int] = None
    delivery_pct: Optional[float] = None

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


class TickData(BaseModel):
    symbol: str
    exchange: Exchange
    timestamp: datetime
    ltp: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: Optional[int] = None
    change_pct: float = 0.0


class InstrumentMeta(BaseModel):
    symbol: str
    exchange: Exchange
    asset_class: AssetClass
    isin: Optional[str] = None
    lot_size: int = 1
    tick_size: float = 0.05
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap_cr: Optional[float] = None
