"""Unit tests for the OpenAI ↔ blockrun-llm boundary."""

from __future__ import annotations

import pytest

from blockrun_litellm import _adapter


def test_sync_round_trip_returns_openai_dict(stub_sync_client) -> None:
    payload = _adapter.chat_completion_sync(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        temperature=0.1,
    )
    assert payload["id"] == "chatcmpl-stub-123"
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "stub-response"
    assert payload["usage"]["total_tokens"] == 15

    stub_sync_client.chat_completion.assert_called_once()
    _, kwargs = stub_sync_client.chat_completion.call_args
    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == 0.1
    # `stream` must not have leaked through (it was never passed).
    assert "stream" not in kwargs


def test_streaming_request_is_rejected(stub_sync_client) -> None:
    with pytest.raises(_adapter.StreamingNotSupported):
        _adapter.chat_completion_sync(
            model="openai/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    stub_sync_client.chat_completion.assert_not_called()


def test_unknown_kwargs_are_dropped(stub_sync_client) -> None:
    """frequency_penalty / logit_bias / etc. should not reach blockrun-llm."""
    _adapter.chat_completion_sync(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        logit_bias={"50256": -100},
        n=1,
    )
    _, kwargs = stub_sync_client.chat_completion.call_args
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs
    assert "logit_bias" not in kwargs
    assert "n" not in kwargs
    assert kwargs["max_tokens"] == 10


def test_tool_calls_are_forwarded(stub_sync_client) -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {}},
        }
    ]
    _adapter.chat_completion_sync(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "weather?"}],
        tools=tools,
        tool_choice="auto",
    )
    _, kwargs = stub_sync_client.chat_completion.call_args
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


def test_blockrun_specific_kwargs_pass_through(stub_sync_client) -> None:
    _adapter.chat_completion_sync(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "latest AI news?"}],
        search=True,
        fallback_models=["openai/gpt-5-mini"],
    )
    _, kwargs = stub_sync_client.chat_completion.call_args
    assert kwargs["search"] is True
    assert kwargs["fallback_models"] == ["openai/gpt-5-mini"]


@pytest.mark.asyncio
async def test_async_round_trip(stub_async_client) -> None:
    payload = await _adapter.chat_completion_async(
        model="anthropic/claude-opus-4-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
    )
    assert payload["model"] == "anthropic/claude-opus-4-5"
    assert payload["choices"][0]["message"]["content"] == "stub-response"
