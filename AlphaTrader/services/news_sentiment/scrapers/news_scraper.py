"""
News scraper: fetches corporate announcements from NSE and article headlines
from financial news aggregators.  Each raw article is published to Kafka
for downstream sentiment processing.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaProducerClient
from shared.schemas.news import RawNewsArticle, NewsSource

log = logging.getLogger(__name__)
settings = get_settings()

NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"
MONEYCONTROL_RSS = "https://www.moneycontrol.com/rss/latestnews.xml"
ET_RSS = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"


class NewsScraper:

    def __init__(self, producer: KafkaProducerClient):
        self._producer = producer
        self._seen_ids: set[str] = set()

    async def run(self) -> None:
        log.info("NewsScraper started")
        while True:
            await asyncio.gather(
                self._fetch_nse_announcements(),
                self._fetch_rss(MONEYCONTROL_RSS, NewsSource.MONEYCONTROL),
                self._fetch_rss(ET_RSS, NewsSource.ECONOMIC_TIMES),
            )
            await asyncio.sleep(120)

    async def _fetch_nse_announcements(self) -> None:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(NSE_ANNOUNCEMENTS_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()
                    for item in (data or [])[:50]:
                        article_id = self._make_id(item.get("desc", "") + item.get("symbol", ""))
                        if article_id in self._seen_ids:
                            continue
                        self._seen_ids.add(article_id)
                        article = RawNewsArticle(
                            article_id=article_id,
                            source=NewsSource.NSE_ANNOUNCEMENTS,
                            headline=item.get("desc", ""),
                            url=f"https://www.nseindia.com{item.get('attchmntFile', '')}",
                            published_at=datetime.utcnow(),
                            scraped_at=datetime.utcnow(),
                            companies_mentioned=[item.get("symbol", "")],
                        )
                        self._producer.publish(settings.kafka.topic_news_raw, article, key=article.companies_mentioned[0])
        except Exception as exc:
            log.exception("NSE announcement fetch error: %s", exc)

    async def _fetch_rss(self, url: str, source: NewsSource) -> None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    text = await resp.text()
            soup = BeautifulSoup(text, "xml")
            for item in soup.find_all("item")[:30]:
                headline = (item.find("title") or {}).get_text(strip=True)
                link = (item.find("link") or {}).get_text(strip=True)
                article_id = self._make_id(headline)
                if article_id in self._seen_ids:
                    continue
                self._seen_ids.add(article_id)
                article = RawNewsArticle(
                    article_id=article_id,
                    source=source,
                    headline=headline,
                    url=link,
                    published_at=datetime.utcnow(),
                    scraped_at=datetime.utcnow(),
                )
                self._producer.publish(settings.kafka.topic_news_raw, article, key="MARKET")
        except Exception as exc:
            log.exception("RSS fetch error for %s: %s", source, exc)

    @staticmethod
    def _make_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
