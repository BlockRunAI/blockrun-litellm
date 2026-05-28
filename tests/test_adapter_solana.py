"""
Unit tests for the Solana branch in ``_adapter.py``.

Adapter v0.3.0 dispatches between ``LLMClient`` (Base) and
``SolanaLLMClient`` (Solana) based on ``api_url``. These tests verify:

- ``api_url`` containing ``sol.blockrun.ai`` routes to ``SolanaLLMClient``.
- A bare or Base-shaped ``api_url`` keeps the default ``LLMClient``.
- ``tools`` / ``tool_choice`` are silently dropped on the Solana path
  (the Solana SDK doesn't accept them).
- Async + Solana raises ``NotImplementedError`` (SDK has no async Solana).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# The solana extras need to be installed for these tests to make sense; skip
# the whole module otherwise so CI without [solana] still passes.
pytest.importorskip("x402")
pytest.importorskip("solders")

from blockrun_litellm import _adapter
from blockrun_litellm._adapter import _filter_kwargs, _is_solana_url


def test_is_solana_url_recognizes_gateway():
    assert _is_solana_url("https://sol.blockrun.ai/api") is True
    assert _is_solana_url("https://sol.blockrun.ai/anything") is True


def test_is_solana_url_rejects_base():
    assert _is_solana_url("https://blockrun.ai/api") is False
    assert _is_solana_url(None) is False
    assert _is_solana_url("") is False


def test_filter_kwargs_keeps_tools_on_both_chains():
    """As of blockrun-llm 0.22.1, ``tools`` / ``tool_choice`` are
    supported on Solana too — the kwarg filter no longer drops them."""
    raw = {
        "max_tokens": 64,
        "tools": [{"type": "function", "function": {"name": "x"}}],
        "tool_choice": "auto",
    }
    for is_solana in (True, False):
        out = _filter_kwargs(raw, is_solana=is_solana)
        assert out["tools"] == raw["tools"], f"tools missing for is_solana={is_solana}"
        assert out["tool_choice"] == "auto"


def test_async_solana_returns_async_solana_client(monkeypatch):
    """v0.3.1+: async Solana is supported — should route to
    :class:`AsyncSolanaLLMClient` (or raise ImportError without [solana])."""
    from blockrun_llm import AsyncSolanaLLMClient
    import unittest.mock as mock

    monkeypatch.setattr(_adapter, "_async_clients", {})

    # Patch the constructor so we don't actually init the x402 SDK / signer.
    with mock.patch.object(AsyncSolanaLLMClient, "__init__", return_value=None):
        client = _adapter.get_async_client(
            api_url="https://sol.blockrun.ai/api",
            private_key="bogus",
        )
    assert isinstance(client, AsyncSolanaLLMClient)


def test_solana_client_routes_through_sync_factory(monkeypatch):
    """``get_sync_client`` with a Solana ``api_url`` should instantiate a
    ``SolanaLLMClient``; we patch its constructor to verify the call."""
    instances: list[Any] = []

    class FakeSolanaClient:
        def __init__(self, *, private_key=None, api_url=None):
            instances.append({"private_key": private_key, "api_url": api_url})

    # Reset the module-level cache so the patched class is used.
    monkeypatch.setattr(_adapter, "_sync_clients", {})
    monkeypatch.setattr(_adapter, "SolanaLLMClient", FakeSolanaClient)
    monkeypatch.setattr(_adapter, "_HAS_SOLANA", True)

    client = _adapter.get_sync_client(
        api_url="https://sol.blockrun.ai/api",
        private_key="bogus-solana-key",
    )
    assert isinstance(client, FakeSolanaClient)
    assert instances == [
        {"private_key": "bogus-solana-key", "api_url": "https://sol.blockrun.ai/api"}
    ]


def test_base_client_still_routes_to_llmclient(monkeypatch):
    """Default / Base ``api_url`` keeps using ``LLMClient`` — making sure
    the Solana branch didn't accidentally swallow everything."""
    from blockrun_llm import LLMClient

    monkeypatch.setattr(_adapter, "_sync_clients", {})
    client = _adapter.get_sync_client(api_url="https://blockrun.ai/api")
    assert isinstance(client, LLMClient)


def test_solana_extras_missing_raises_import_error(monkeypatch):
    """If a user sets a Solana ``api_url`` without installing the
    ``[solana]`` extra, we surface a clear ``ImportError`` rather than
    crashing later in the SDK."""
    monkeypatch.setattr(_adapter, "_sync_clients", {})
    monkeypatch.setattr(_adapter, "_HAS_SOLANA", False)
    monkeypatch.setattr(_adapter, "SolanaLLMClient", None)

    with pytest.raises(ImportError, match=r"\[solana\]"):
        _adapter.get_sync_client(api_url="https://sol.blockrun.ai/api")


# ---------------------------------------------------------------------------
# Image generation — Solana branch (added 0.3.8 to fix the
# ``transaction_simulation_failed`` regression for gpt-image-2 et al.)
# ---------------------------------------------------------------------------


def test_image_client_routes_to_solana(monkeypatch):
    """A Solana ``api_url`` should make ``get_image_client`` return a
    ``SolanaLLMClient`` instead of the EVM-only ``ImageClient``."""
    instances: list[Any] = []

    class FakeSolanaClient:
        def __init__(self, *, private_key=None, api_url=None):
            instances.append({"private_key": private_key, "api_url": api_url})

    monkeypatch.setattr(_adapter, "_image_clients", {})
    monkeypatch.setattr(_adapter, "SolanaLLMClient", FakeSolanaClient)
    monkeypatch.setattr(_adapter, "_HAS_SOLANA", True)

    client = _adapter.get_image_client(
        api_url="https://sol.blockrun.ai/api",
        private_key="bogus-solana-key",
    )
    assert isinstance(client, FakeSolanaClient)
    assert instances == [
        {"private_key": "bogus-solana-key", "api_url": "https://sol.blockrun.ai/api"}
    ]


def test_image_client_base_still_uses_imageclient(monkeypatch):
    """Default / Base ``api_url`` keeps using the EVM ``ImageClient``."""
    from blockrun_llm import ImageClient

    monkeypatch.setattr(_adapter, "_image_clients", {})
    client = _adapter.get_image_client(api_url="https://blockrun.ai/api")
    assert isinstance(client, ImageClient)


def test_image_client_solana_missing_extras_raises(monkeypatch):
    """Solana ``api_url`` without [solana] extras → clear ImportError."""
    monkeypatch.setattr(_adapter, "_image_clients", {})
    monkeypatch.setattr(_adapter, "_HAS_SOLANA", False)
    monkeypatch.setattr(_adapter, "SolanaLLMClient", None)

    with pytest.raises(ImportError, match=r"\[solana\]"):
        _adapter.get_image_client(api_url="https://sol.blockrun.ai/api")


def test_image_generation_sync_dispatches_to_solana_image(monkeypatch):
    """``image_generation_sync`` on Solana should call ``.image(...)`` on
    ``SolanaLLMClient`` (not ``.generate(...)`` which only exists on the
    EVM ``ImageClient``)."""
    captured: dict[str, Any] = {}

    class FakeResponse:
        def model_dump(self, exclude_none=True):
            return {"data": [{"url": "https://example/img.png"}]}

    class FakeSolanaClient:
        def __init__(self, *, private_key=None, api_url=None):
            pass

        def image(self, prompt, *, model=None, size=None, n=1):
            captured.update(prompt=prompt, model=model, size=size, n=n)
            return FakeResponse()

    monkeypatch.setattr(_adapter, "_image_clients", {})
    monkeypatch.setattr(_adapter, "SolanaLLMClient", FakeSolanaClient)
    monkeypatch.setattr(_adapter, "_HAS_SOLANA", True)

    out = _adapter.image_generation_sync(
        "a red apple",
        model="openai/gpt-image-2",
        size="1024x1024",
        n=1,
        api_url="https://sol.blockrun.ai/api",
        private_key="bogus",
    )
    assert out == {"data": [{"url": "https://example/img.png"}]}
    assert captured == {
        "prompt": "a red apple",
        "model": "openai/gpt-image-2",
        "size": "1024x1024",
        "n": 1,
    }
