import httpx
import pytest
from fastapi.testclient import TestClient

from ollama_router.api import create_app
from ollama_router.config import AppConfig


def make_config():
    return AppConfig.model_validate(
        {
            "healthcheck_interval_seconds": 9999,
            "healthcheck_timeout_seconds": 1,
            "request_timeout_seconds": 5,
            "backends": [
                {"name": "primary", "priority": 10, "endpoint": "http://primary.test"},
                {"name": "fallback", "priority": 20, "endpoint": "http://fallback.test", "api_key": "secret"},
            ],
        }
    )


def test_health():
    async def handler(request):
        return httpx.Response(200, json={"models": []})

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(make_config(), client=httpx.AsyncClient(transport=transport))) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_status_lists_backends_with_last_seen():
    async def handler(request):
        return httpx.Response(200, json={"models": []})

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(make_config(), client=httpx.AsyncClient(transport=transport))) as client:
        response = client.get("/status")
        assert response.status_code == 200
        body = response.json()
        assert [backend["name"] for backend in body["backends"]] == ["primary", "fallback"]
        assert body["backends"][0]["healthy"] is True
        assert body["backends"][0]["last_seen"] is not None


def test_chat_routes_to_first_healthy_priority_backend():
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url), request.headers.get("authorization")))
        if str(request.url) == "http://primary.test/api/tags":
            return httpx.Response(200, json={"models": []})
        if str(request.url) == "http://fallback.test/api/tags":
            return httpx.Response(200, json={"models": []})
        if str(request.url) == "http://primary.test/api/chat":
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:0.5b",
                    "message": {"role": "assistant", "content": "ok from primary"},
                    "prompt_eval_count": 1,
                    "eval_count": 2,
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(make_config(), client=httpx.AsyncClient(transport=transport))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["router"]["backend"] == "primary"
        assert body["choices"][0]["message"]["content"] == "ok from primary"
        assert ("POST", "http://primary.test/api/chat", None) in calls


def test_chat_falls_back_when_primary_unhealthy():
    async def handler(request):
        if str(request.url) == "http://primary.test/api/tags":
            return httpx.Response(503)
        if str(request.url) == "http://fallback.test/api/tags":
            return httpx.Response(200, json={"models": []})
        if str(request.url) == "http://fallback.test/api/chat":
            assert request.headers["authorization"] == "Bearer secret"
            return httpx.Response(200, json={"model": "m", "message": {"content": "fallback"}})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(make_config(), client=httpx.AsyncClient(transport=transport))) as client:
        response = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200
        assert response.json()["router"]["backend"] == "fallback"


def test_chat_returns_503_when_no_backend_healthy():
    async def handler(request):
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(make_config(), client=httpx.AsyncClient(transport=transport))) as client:
        response = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 503


def test_streaming_request_returns_400():
    async def handler(request):
        return httpx.Response(200, json={"models": []})

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(make_config(), client=httpx.AsyncClient(transport=transport))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert response.status_code == 400
