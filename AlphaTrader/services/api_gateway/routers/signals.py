"""
Signal endpoints — query historical and live signals.
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from pydantic import BaseModel
from shared.schemas.signal import TradingSignal, SignalType, TradingStyle
from ..auth.dependencies import get_current_user

router = APIRouter()


class SignalListResponse(BaseModel):
    signals: list[TradingSignal]
    count: int
    page: int
    page_size: int


@router.get("/", response_model=SignalListResponse)
async def list_signals(
    symbol: Optional[str] = Query(None),
    signal_type: Optional[SignalType] = Query(None),
    trading_style: Optional[TradingStyle] = Query(None),
    min_confidence: float = Query(default=0.60, ge=0.0, le=1.0),
    from_dt: Optional[datetime] = Query(default=None),
    to_dt: Optional[datetime] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    _user=Depends(get_current_user),
):
    """
    Query generated trading signals with optional filters.
    Returns paginated results sorted by generated_at DESC.
    """
    # TODO: replace with actual TimescaleDB query
    return SignalListResponse(signals=[], count=0, page=page, page_size=page_size)


@router.get("/{signal_id}", response_model=TradingSignal)
async def get_signal(signal_id: str, _user=Depends(get_current_user)):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Signal not found")


@router.get("/summary/today")
async def today_summary(_user=Depends(get_current_user)):
    """Returns a count breakdown of today's signals by type."""
    return {
        "date": datetime.utcnow().date().isoformat(),
        "STRONG_BUY": 0,
        "BUY": 0,
        "SELL": 0,
        "STRONG_SELL": 0,
        "HOLD": 0,
    }
