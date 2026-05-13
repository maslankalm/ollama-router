import os

import pytest

from ollama_router.config import AppConfig, load_config


def test_config_sorts_by_priority():
    config = AppConfig.model_validate(
        {
            "backends": [
                {"name": "fallback", "priority": 100, "endpoint": "http://fallback:11434/"},
                {"name": "primary", "priority": 10, "endpoint": "http://primary:11434"},
            ]
        }
    )
    assert [backend.name for backend in config.backends] == ["primary", "fallback"]
    assert config.backends[1].endpoint == "http://fallback:11434"


def test_duplicate_backend_names_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "backends": [
                    {"name": "same", "priority": 1, "endpoint": "http://a:11434"},
                    {"name": "same", "priority": 2, "endpoint": "http://b:11434"},
                ]
            }
        )


def test_api_key_env_wins(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_KEY", "from-env")
    config = AppConfig.model_validate(
        {"backends": [{"name": "cloud", "priority": 1, "endpoint": "https://ollama.com", "api_key": "fallback", "api_key_env": "TEST_OLLAMA_KEY"}]}
    )
    assert config.backends[0].resolved_api_key() == "from-env"


def test_load_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("backends:\n  - name: one\n    priority: 1\n    endpoint: http://one:11434\n")
    assert load_config(path).backends[0].name == "one"
