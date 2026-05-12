"""
Shared adapter between OpenAI-format requests and the blockrun-llm SDK.

Used by both the LiteLLM CustomLLM provider (in-process) and the FastAPI
proxy (sidecar). Keeping the conversion logic in one place ensures the two
modes behave identically.

Design notes
------------
- BlockRun's HTTP API is already OpenAI-shaped. The blockrun-llm SDK
  returns a Pydantic ``ChatResponse`` (or ``ChatCompletionChunk`` in
  stream mode) whose ``.model_dump()`` is a valid OpenAI Chat Completions
  response (or chunk).
- We therefore only translate at the *boundary*: accept an OpenAI dict
  request, dispatch through ``LLMClient.chat_completion(...)`` /
  ``chat_completion_stream(...)`` so x402 signing happens inside the SDK,
  and return the dumped pydantic models.
- The ``model`` string is forwarded verbatim. LiteLLM strips its
  ``blockrun/`` prefix before invoking the handler, so values like
  ``openai/gpt-5.5`` reach this layer unchanged — which is exactly what
  the BlockRun gateway expects.
"""

from __future__ import annotations

import os
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from blockrun_llm import AsyncLLMClient, LLMClient
from blockrun_llm.types import APIError, ChatCompletionChunk, PaymentError


# ---------------------------------------------------------------------------
# Client cache
# ---------------------------------------------------------------------------
# Constructing an LLMClient parses the private key and instantiates an HTTP
# session. We memoize per (api_url, private_key) so high-QPS adapters don't
# re-create wallets for every request.

_sync_clients: Dict[str, LLMClient] = {}
_async_clients: Dict[str, AsyncLLMClient] = {}
_lock = threading.Lock()


def _client_key(api_url: Optional[str], private_key: Optional[str]) -> str:
    return f"{api_url or ''}::{private_key or os.environ.get('BLOCKRUN_WALLET_KEY', '')}"


def get_sync_client(
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> LLMClient:
    """Return a cached sync ``LLMClient`` for the given creds/url."""
    key = _client_key(api_url, private_key)
    with _lock:
        client = _sync_clients.get(key)
        if client is None:
            client = LLMClient(private_key=private_key, api_url=api_url)
            _sync_clients[key] = client
        return client


def get_async_client(
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> AsyncLLMClient:
    """Return a cached async ``AsyncLLMClient`` for the given creds/url."""
    key = _client_key(api_url, private_key)
    with _lock:
        client = _async_clients.get(key)
        if client is None:
            client = AsyncLLMClient(private_key=private_key, api_url=api_url)
            _async_clients[key] = client
        return client


# ---------------------------------------------------------------------------
# Request normalization
# ---------------------------------------------------------------------------

# OpenAI-style params that blockrun-llm's chat methods accept directly.
# Anything outside this set is dropped — LiteLLM tends to forward
# provider-specific kwargs that don't apply here.
_FORWARDED_KWARGS = {
    "max_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "search",
    "search_parameters",
    "fallback_models",
}


def _filter_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist OpenAI-format kwargs into the shape blockrun-llm wants.

    Does **not** raise on ``stream=True`` — streaming has its own entrypoint.
    """
    return {k: payload[k] for k in _FORWARDED_KWARGS if payload.get(k) is not None}


# ---------------------------------------------------------------------------
# Non-streaming entrypoints
# ---------------------------------------------------------------------------


def chat_completion_sync(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a non-streaming chat completion via blockrun-llm; return an
    OpenAI-format dict.

    ``model`` is forwarded as-is to the BlockRun gateway (e.g.
    ``"openai/gpt-5.5"``, ``"anthropic/claude-opus-4-5"``). Any LiteLLM
    ``blockrun/`` prefix should be stripped by the caller.

    For ``stream=True``, use :func:`chat_completion_stream_sync` instead —
    LiteLLM dispatches streaming through a different entrypoint.
    """
    # Ignore stream= if it leaks in here; callers should route through the
    # stream functions for the streaming case.
    openai_kwargs.pop("stream", None)
    kwargs = _filter_kwargs(openai_kwargs)
    client = get_sync_client(api_url=api_url, private_key=private_key)
    response = client.chat_completion(model=model, messages=messages, **kwargs)
    return response.model_dump(exclude_none=True)


async def chat_completion_async(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> Dict[str, Any]:
    """Async variant of :func:`chat_completion_sync`."""
    openai_kwargs.pop("stream", None)
    kwargs = _filter_kwargs(openai_kwargs)
    client = get_async_client(api_url=api_url, private_key=private_key)
    response = await client.chat_completion(model=model, messages=messages, **kwargs)
    return response.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Streaming entrypoints
# ---------------------------------------------------------------------------


def chat_completion_stream_sync(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> Iterator[ChatCompletionChunk]:
    """
    Stream a chat completion via blockrun-llm SDK's
    :meth:`LLMClient.chat_completion_stream`.

    Yields :class:`ChatCompletionChunk` objects (OpenAI ``chat.completion.chunk``
    schema). Caller is responsible for downstream formatting — e.g. the
    LiteLLM provider maps each chunk into a ``GenericStreamingChunk``, while
    the FastAPI proxy serializes each as ``data: <json>\\n\\n``.
    """
    openai_kwargs.pop("stream", None)
    kwargs = _filter_kwargs(openai_kwargs)
    client = get_sync_client(api_url=api_url, private_key=private_key)
    yield from client.chat_completion_stream(model=model, messages=messages, **kwargs)


async def chat_completion_stream_async(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> AsyncIterator[ChatCompletionChunk]:
    """Async variant of :func:`chat_completion_stream_sync`."""
    openai_kwargs.pop("stream", None)
    kwargs = _filter_kwargs(openai_kwargs)
    client = get_async_client(api_url=api_url, private_key=private_key)
    async for chunk in client.chat_completion_stream(
        model=model, messages=messages, **kwargs
    ):
        yield chunk


__all__ = [
    "chat_completion_sync",
    "chat_completion_async",
    "chat_completion_stream_sync",
    "chat_completion_stream_async",
    "get_sync_client",
    "get_async_client",
    "APIError",
    "PaymentError",
]
