"""
Shared test fixtures.

We never hit the real BlockRun gateway or the EVM in unit tests — every
test stubs out ``blockrun_litellm._adapter.get_sync_client`` /
``get_async_client`` with a fake that returns a canned ``ChatResponse``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from blockrun_llm.types import ChatChoice, ChatMessage, ChatResponse, ChatUsage

from blockrun_litellm import _adapter, _apikey


# ---------------------------------------------------------------------------
# Canned response builder
# ---------------------------------------------------------------------------

def make_chat_response(
    *,
    model: str = "openai/gpt-5.5",
    content: str = "stub-response",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-stub-123",
        object="chat.completion",
        created=1_700_000_000,
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


# ---------------------------------------------------------------------------
# Auto-patch the client cache so no real wallet is ever needed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_wallet_required(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    """Run every test on a host with no credentials and no chain preference.

    Since 0.10.0 the implicit chain default reads ``~/.blockrun`` twice — for a
    chain the CLI recorded, and for which wallets exist. Left alone, the suite
    would resolve a different default on a developer's laptop than in CI, which
    is exactly the kind of test that passes everywhere except where it matters.
    (It bit immediately: this machine has ``~/.blockrun/.chain`` set to "base".)
    So both the session files and the chain files are pointed at a directory
    that does not exist, the credential env vars are cleared, and the memoized
    answer is dropped: every test starts on the documented default (Solana)
    unless it says otherwise.
    """
    for name in (
        "BLOCKRUN_WALLET_KEY",
        "BASE_CHAIN_WALLET_KEY",
        "SOLANA_WALLET_KEY",
        "BLOCKRUN_API_KEY",
        "BLOCKRUN_CHAIN",
        "BLOCKRUN_API_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    absent = tmp_path_factory.mktemp("no-wallets") / "nowhere"
    monkeypatch.setattr(_adapter, "_SOLANA_SESSION", absent / ".solana-session")
    monkeypatch.setattr(_adapter, "_BASE_SESSION", absent / ".session")
    monkeypatch.setattr(
        _adapter, "_CHAIN_FILES", (absent / "payment-chain", absent / ".chain")
    )
    _adapter._reset_chain_cache_for_tests()
    _apikey._reset_clients_for_tests()


@pytest.fixture
def stub_sync_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the cached sync client with a MagicMock returning canned data."""
    mock = MagicMock()
    mock.chat_completion.return_value = make_chat_response()

    def _get(api_url=None, private_key=None):  # noqa: ANN001 - test stub
        return mock

    monkeypatch.setattr("blockrun_litellm._adapter.get_sync_client", _get)
    return mock


@pytest.fixture
def stub_async_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the cached async client with an AsyncMock-style stub."""
    mock = MagicMock()

    async def _chat_completion(model, messages, **kwargs):  # noqa: ANN001
        return make_chat_response(model=model)

    mock.chat_completion = _chat_completion

    def _get(api_url=None, private_key=None):  # noqa: ANN001 - test stub
        return mock

    monkeypatch.setattr("blockrun_litellm._adapter.get_async_client", _get)
    return mock
