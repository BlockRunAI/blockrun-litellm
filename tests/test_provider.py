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
        model="anthropic/claude-opus-4-7",
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


# ---------------------------------------------------------------------------
# Exception translation — transient BlockRun errors should surface as
# LiteLLM's retriable exception types so the router can fall back.
# ---------------------------------------------------------------------------

class TestExceptionTranslation:
    """Maps in :func:`blockrun_litellm.provider._translate_to_litellm`."""

    def test_503_becomes_service_unavailable(self, monkeypatch):
        from blockrun_llm.types import APIError as BRError

        def _raise(*args, **kwargs):
            raise BRError("upstream down", 503, {"error": "down"})

        # Patch the adapter so handler.completion fires our APIError.
        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(litellm.ServiceUnavailableError):
            handler.completion(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_502_becomes_api_connection_error(self, monkeypatch):
        from blockrun_llm.types import APIError as BRError

        def _raise(*args, **kwargs):
            raise BRError("bad gateway", 502, {"error": "bad"})

        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(litellm.APIConnectionError):
            handler.completion(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_500_becomes_internal_server_error(self, monkeypatch):
        from blockrun_llm.types import APIError as BRError

        def _raise(*args, **kwargs):
            raise BRError("oops", 500, {"error": "oops"})

        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(litellm.InternalServerError):
            handler.completion(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_429_becomes_rate_limit_error(self, monkeypatch):
        from blockrun_llm.types import APIError as BRError

        def _raise(*args, **kwargs):
            raise BRError("slow down", 429, {"error": "rate_limit"})

        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(litellm.RateLimitError):
            handler.completion(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_timeout_becomes_litellm_timeout(self, monkeypatch):
        import httpx as _httpx

        def _raise(*args, **kwargs):
            raise _httpx.ConnectTimeout("connect timed out")

        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(litellm.Timeout):
            handler.completion(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_non_transient_errors_pass_through(self, monkeypatch):
        """A plain RuntimeError (or any non-transient exc) should NOT be
        translated — propagate as-is so the caller sees the real error."""

        def _raise(*args, **kwargs):
            raise RuntimeError("config bug")

        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(RuntimeError, match="config bug"):
            handler.completion(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_stream_503_translates(self, monkeypatch):
        """Stream-side: SDK raises APIError(503) → LiteLLM sees
        ServiceUnavailableError before any chunks are yielded."""
        from blockrun_llm.types import APIError as BRError

        def _raise(*args, **kwargs):
            raise BRError("upstream down", 503, {"error": "down"})
            yield  # make it a generator

        monkeypatch.setattr(
            "blockrun_litellm._adapter.chat_completion_stream_sync",
            _raise,
        )
        handler = BlockRunLLM()
        with pytest.raises(litellm.ServiceUnavailableError):
            list(handler.streaming(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            ))
