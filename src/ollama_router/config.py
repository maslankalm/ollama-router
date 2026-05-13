from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendConfig(BaseModel):
    name: str
    priority: int = Field(ge=0)
    endpoint: str
    api_key: str | None = None
    api_key_env: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or any(char.isspace() for char in value):
            raise ValueError("backend name must be non-empty and contain no whitespace")
        return value

    @field_validator("endpoint")
    @classmethod
    def normalize_endpoint(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("endpoint must start with http:// or https://")
        return value

    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.getenv(self.api_key_env) or self.api_key
        return self.api_key


class AppConfig(BaseModel):
    healthcheck_interval_seconds: float = Field(default=30, gt=0)
    healthcheck_timeout_seconds: float = Field(default=5, gt=0)
    request_timeout_seconds: float = Field(default=120, gt=0)
    backends: list[BackendConfig]

    @field_validator("backends")
    @classmethod
    def validate_backends(cls, value: list[BackendConfig]) -> list[BackendConfig]:
        if not value:
            raise ValueError("at least one backend is required")
        names = [backend.name for backend in value]
        if len(names) != len(set(names)):
            raise ValueError("backend names must be unique")
        return sorted(value, key=lambda backend: (backend.priority, backend.name))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_ROUTER_", extra="ignore")

    config_path: Path = Path("/etc/ollama-router/config.yaml")
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    data: Any = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration file must contain a YAML object")
    return AppConfig.model_validate(data)
