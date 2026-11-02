import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def api_client():
    from services.api_gateway.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers(api_client):
    """Returns valid JWT auth headers for test requests."""
    import asyncio
    response = asyncio.get_event_loop().run_until_complete(
        api_client.post(
            "/api/v1/auth/token",
            data={"username": "demo@alphatrader.app", "password": "demo1234"},
        )
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
