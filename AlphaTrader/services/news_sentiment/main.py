"""
News sentiment service entrypoint.
Runs scraper and FinBERT analyzer as concurrent tasks.
"""

import asyncio
import logging
import threading
from shared.messaging.kafka_client import KafkaProducerClient
from services.news_sentiment.scrapers.news_scraper import NewsScraper
from services.news_sentiment.nlp.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)


async def main():
    producer = KafkaProducerClient()
    scraper = NewsScraper(producer)

    # SentimentAnalyzer runs a blocking consumer loop — run in thread
    analyzer = SentimentAnalyzer()
    analyzer_thread = threading.Thread(target=analyzer.run, daemon=True)
    analyzer_thread.start()

    log.info("News scraper starting...")
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
