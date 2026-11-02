"""
Feature engine service entrypoint.
Runs the streaming feature pipeline as a long-lived process.
"""

import asyncio
import logging
from services.feature_engine.pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)


async def main():
    pipeline = FeaturePipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
