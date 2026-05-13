from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from .api import create_app
from .config import Settings, load_config


def build_app() -> FastAPI:
    settings = Settings()
    config = load_config(settings.config_path)
    return create_app(config)


def run() -> None:
    settings = Settings()
    config = load_config(settings.config_path)
    uvicorn.run(create_app(config), host=settings.host, port=settings.port, log_level=settings.log_level)


_app: FastAPI | None = None


class _LazyApp:
    def __getattr__(self, name):
        global _app
        if _app is None:
            _app = build_app()
        return getattr(_app, name)

    async def __call__(self, scope, receive, send):
        global _app
        if _app is None:
            _app = build_app()
        await _app(scope, receive, send)


app = _LazyApp()


if __name__ == "__main__":
    run()
