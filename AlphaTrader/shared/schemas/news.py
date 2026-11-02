from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NewsSource(str, Enum):
    MONEYCONTROL = "moneycontrol"
    ECONOMIC_TIMES = "economic_times"
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    LIVEMINT = "livemint"
    BUSINESS_STANDARD = "business_standard"
    NSE_ANNOUNCEMENTS = "nse_announcements"
    TWITTER = "twitter"
    REDDIT = "reddit"


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class RawNewsArticle(BaseModel):
    article_id: str
    source: NewsSource
    headline: str
    body: Optional[str] = None
    url: str
    published_at: datetime
    scraped_at: datetime
    companies_mentioned: list[str] = Field(default_factory=list)
    sectors_mentioned: list[str] = Field(default_factory=list)


class SentimentResult(BaseModel):
    article_id: str
    symbol: Optional[str] = None
    sector: Optional[str] = None
    label: SentimentLabel
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)
    composite_score: float = Field(..., ge=-1.0, le=1.0, description="-1 very negative, +1 very positive")
    impact_level: str = Field(default="LOW", description="LOW | MEDIUM | HIGH")
    model_used: str = "finbert"
    analyzed_at: datetime
