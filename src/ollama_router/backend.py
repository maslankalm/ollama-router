from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import AppConfig, BackendConfig


@dataclass
class BackendState:
    name: str
    priority: int
    endpoint: str
    healthy: bool = False
    last_checked: float | None = None
    last_seen: float | None = None
    detail: str = "not checked"


class BackendPool:
    def __init__(self, config: AppConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._states = {
            backend.name: BackendState(name=backend.name, priority=backend.priority, endpoint=backend.endpoint)
            for backend in config.backends
        }
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def close(self) -> None:
        await self.stop()
        if self._owns_client:
            await self.client.aclose()

    async def start(self) -> None:
        await self.check_all()
        self._task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.healthcheck_interval_seconds)
            await self.check_all()

    async def check_all(self) -> None:
        await asyncio.gather(*(self.check_backend(backend) for backend in self.config.backends))

    async def check_backend(self, backend: BackendConfig) -> BackendState:
        checked_at = time.time()
        healthy = False
        detail = "unhealthy"
        try:
            response = await self.client.get(
                f"{backend.endpoint}/api/tags",
                headers=_headers(backend),
                timeout=self.config.healthcheck_timeout_seconds,
            )
            response.raise_for_status()
            healthy = True
            detail = "ok"
        except Exception as exc:  # health detail is operational, not user input
            detail = f"{type(exc).__name__}: {exc}"

        async with self._lock:
            state = self._states[backend.name]
            state.healthy = healthy
            state.last_checked = checked_at
            state.detail = detail
            if healthy:
                state.last_seen = checked_at
            return BackendState(**state.__dict__)

    async def status(self) -> list[BackendState]:
        async with self._lock:
            return [BackendState(**self._states[backend.name].__dict__) for backend in self.config.backends]

    async def first_healthy(self) -> BackendConfig | None:
        async with self._lock:
            healthy_names = {name for name, state in self._states.items() if state.healthy}
        for backend in self.config.backends:
            if backend.name in healthy_names:
                return backend
        await self.check_all()
        async with self._lock:
            healthy_names = {name for name, state in self._states.items() if state.healthy}
        for backend in self.config.backends:
            if backend.name in healthy_names:
                return backend
        return None

    async def chat(self, payload: dict[str, Any]) -> tuple[BackendConfig, dict[str, Any]]:
        backend = await self.first_healthy()
        if backend is None:
            raise NoHealthyBackendError("no healthy Ollama backend available")
        try:
            response = await self.client.post(
                f"{backend.endpoint}/api/chat",
                headers=_headers(backend),
                json=payload,
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            return backend, response.json()
        except Exception:
            await self.check_backend(backend)
            raise


def _headers(backend: BackendConfig) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = backend.resolved_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class NoHealthyBackendError(RuntimeError):
    pass
