from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from .backend import BackendPool, NoHealthyBackendError
from .config import AppConfig
from .openai import ChatCompletionRequest, from_ollama_response, to_ollama_payload


class HealthResponse(BaseModel):
    status: str


class BackendStatus(BaseModel):
    name: str
    priority: int
    endpoint: str
    healthy: bool
    last_checked: float | None
    last_seen: float | None
    detail: str


class StatusResponse(BaseModel):
    backends: list[BackendStatus]


def create_app(config: AppConfig, client: httpx.AsyncClient | None = None) -> FastAPI:
    pool = BackendPool(config=config, client=client)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await pool.start()
        try:
            yield
        finally:
            await pool.close()

    app = FastAPI(title="Ollama Router", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/status", response_model=StatusResponse)
    async def router_status() -> StatusResponse:
        states = await pool.status()
        return StatusResponse(backends=[BackendStatus(**state.__dict__) for state in states])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict:
        try:
            payload = to_ollama_payload(request)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        try:
            backend, response = await pool.chat(payload)
        except NoHealthyBackendError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"backend returned HTTP {exc.response.status_code}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"backend request failed: {exc}") from exc

        return from_ollama_response(request.model, backend.name, response)

    return app
