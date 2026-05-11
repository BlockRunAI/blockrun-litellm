"""Smoke tests for the LiteLLM CustomLLM handler."""

from __future__ import annotations

import litellm
import pytest

from blockrun_litellm import BlockRunLLM, register


def test_register_is_idempotent() -> None:
    handler1 = register()
    handler2 = register()
    assert handler1 is handler2

    entries = [
        e for e in (litellm.custom_provider_map or [])
        if isinstance(e, dict) and e.get("provider") == "blockrun"
    ]
    assert len(entries) == 1


def test_completion_returns_model_response(stub_sync_client) -> None:
    handler = BlockRunLLM()
    response = handler.completion(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 32, "temperature": 0.0},
    )
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "stub-response"


@pytest.mark.asyncio
async def test_acompletion_returns_model_response(stub_async_client) -> None:
    handler = BlockRunLLM()
    response = await handler.acompletion(
        model="anthropic/claude-opus-4-5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 16},
    )
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "stub-response"


def test_api_base_and_api_key_forwarded(stub_sync_client, monkeypatch) -> None:
    """LiteLLM passes api_base/api_key through to the handler. They should
    reach the SDK as api_url / private_key respectively."""
    captured = {}

    def _fake_get(api_url=None, private_key=None):  # noqa: ANN001
        captured["api_url"] = api_url
        captured["private_key"] = private_key
        return stub_sync_client

    monkeypatch.setattr("blockrun_litellm._adapter.get_sync_client", _fake_get)

    handler = BlockRunLLM()
    handler.completion(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://gateway.example.test/api",
        api_key="0xDEADBEEF" + "00" * 28,
    )

    assert captured["api_url"] == "https://gateway.example.test/api"
    assert captured["private_key"].startswith("0xDEADBEEF")
