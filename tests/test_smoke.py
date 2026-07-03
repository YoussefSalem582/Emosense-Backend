"""
End-to-end smoke test for the EmoSense API.

Exercises the core authenticated flow against the real ASGI app on SQLite:
register -> login -> analyze text -> auth enforcement -> list -> get-by-id ->
batch -> refresh -> me. Text analysis uses the keyword fallback, so this runs
without the heavy ML stack or a model download, making it a fast, reliable CI gate.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_end_to_end_flow(async_client: AsyncClient):
    # Register
    reg = {
        "email": "smoke@example.com",
        "password": "SecurePass123!",
        "confirm_password": "SecurePass123!",
        "first_name": "Smoke",
        "last_name": "Test",
    }
    r = await async_client.post("/api/v1/auth/register", json=reg)
    assert r.status_code == 201, r.text
    assert r.json()["email"] == reg["email"]

    # Login
    r = await async_client.post(
        "/api/v1/auth/login",
        json={"email": reg["email"], "password": reg["password"]},
    )
    assert r.status_code == 200, r.text
    tokens = r.json()
    assert tokens["access_token"] and tokens["refresh_token"]
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}

    # Authenticated text analysis
    r = await async_client.post(
        "/api/v1/emotion/text",
        json={"text": "I am so happy and excited, this is wonderful!", "segment_analysis": False},
        headers=auth,
    )
    assert r.status_code == 201, r.text
    analysis = r.json()
    analysis_id = analysis["id"]
    assert analysis["dominant_emotion"] is not None

    # Auth is enforced
    r = await async_client.post("/api/v1/emotion/text", json={"text": "hello"})
    assert r.status_code in (401, 403)

    # List
    r = await async_client.get("/api/v1/emotion/", headers=auth)
    assert r.status_code == 200, r.text
    assert len(r.json()) >= 1

    # Get by id
    r = await async_client.get(f"/api/v1/emotion/{analysis_id}", headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == analysis_id

    # Batch
    r = await async_client.post(
        "/api/v1/emotion/batch",
        json={"analysis_type": "text", "inputs": ["I feel sad", "totally neutral"]},
        headers=auth,
    )
    assert r.status_code == 202, r.text
    assert r.json()["total_items"] == 2

    # Refresh
    r = await async_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert r.status_code == 200, r.text

    # Current user
    r = await async_client.get("/api/v1/users/me", headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["email"] == reg["email"]


async def test_health_endpoint(async_client: AsyncClient):
    r = await async_client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()
