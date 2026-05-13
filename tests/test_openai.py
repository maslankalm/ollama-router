import pytest

from ollama_router.openai import ChatCompletionRequest, from_ollama_response, to_ollama_payload


def test_to_ollama_payload_maps_basic_chat():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen2.5:0.5b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "max_tokens": 12,
        }
    )
    payload = to_ollama_payload(request)
    assert payload == {
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 12},
    }


def test_streaming_rejected_for_now():
    request = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": True}
    )
    with pytest.raises(ValueError):
        to_ollama_payload(request)


def test_from_ollama_response_openai_shape():
    result = from_ollama_response(
        "requested-model",
        "primary",
        {"model": "actual-model", "message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": 3, "eval_count": 2},
    )
    assert result["object"] == "chat.completion"
    assert result["model"] == "actual-model"
    assert result["router"] == {
        "backend": "primary",
        "requested_model": "requested-model",
        "response_model": "actual-model",
    }
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "ok"}
    assert result["usage"]["total_tokens"] == 5
