from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from shared.config.settings import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.db.timescale_url,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
    echo=settings.environment == "development",
)

AsyncSessionFactory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db_session():
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
