from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    response_format: dict[str, Any] | None = None


def to_ollama_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise ValueError("streaming responses are not implemented in this minimal router")
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [message.model_dump() for message in request.messages],
        "stream": False,
    }
    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.response_format and request.response_format.get("type") == "json_object":
        payload["format"] = "json"
    if options:
        payload["options"] = options
    return payload


def from_ollama_response(model: str, backend_name: str, data: dict[str, Any]) -> dict[str, Any]:
    message = data.get("message") or {}
    content = message.get("content") or data.get("response") or ""
    response_model = data.get("model") or model
    prompt_tokens = data.get("prompt_eval_count") or 0
    completion_tokens = data.get("eval_count") or 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response_model,
        "router": {
            "backend": backend_name,
            "requested_model": model,
            "response_model": response_model,
        },
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if data.get("done", True) else None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
