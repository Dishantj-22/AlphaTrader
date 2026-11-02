import pytest


class TestAuthEndpoints:

    @pytest.mark.asyncio
    async def test_login_valid_credentials(self, api_client):
        response = await api_client.post(
            "/api/v1/auth/token",
            data={"username": "demo@alphatrader.app", "password": "demo1234"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, api_client):
        response = await api_client.post(
            "/api/v1/auth/token",
            data={"username": "bad@user.com", "password": "wrong"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_without_token_is_401(self, api_client):
        response = await api_client.get("/api/v1/auth/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        response = await api_client.get("/internal/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_signals_requires_auth(self, api_client):
        response = await api_client.get("/api/v1/signals/")
        assert response.status_code == 401
