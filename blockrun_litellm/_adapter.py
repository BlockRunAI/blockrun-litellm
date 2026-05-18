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

import asyncio
import os
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from blockrun_llm import AsyncLLMClient, ImageClient, LLMClient
from blockrun_llm.types import APIError, ChatCompletionChunk, PaymentError

try:
    from blockrun_llm import AsyncSolanaLLMClient, SolanaLLMClient

    _HAS_SOLANA = True
except ImportError:
    _HAS_SOLANA = False
    SolanaLLMClient = None  # type: ignore[assignment]
    AsyncSolanaLLMClient = None  # type: ignore[assignment]


# Default endpoints — used to decide chain when no explicit api_url is given.
SOLANA_API_URL = "https://sol.blockrun.ai/api"
BASE_API_URL = "https://blockrun.ai/api"


def _is_solana_url(api_url: Optional[str]) -> bool:
    """Sniff whether the effective gateway URL points at Solana.

    Falls back to the ``BLOCKRUN_API_URL`` env var when no explicit
    ``api_url`` is passed. This matters for the FastAPI sidecar: the
    request handlers don't forward an ``api_url`` arg, so without the
    env-var fallback we'd silently route Solana traffic to the Base
    async client and crash inside the EVM payment encoder
    (``eth_abi.AddressEncoder`` rejects base58 mint addresses).
    """
    resolved = api_url or os.environ.get("BLOCKRUN_API_URL", "")
    return bool(resolved) and "sol.blockrun.ai" in resolved


# ---------------------------------------------------------------------------
# Client cache (Base + Solana)
# ---------------------------------------------------------------------------
# Constructing a client parses the private key and instantiates an HTTP
# session. We memoize per (chain, api_url, private_key) so high-QPS adapters
# don't re-create wallets for every request. The chain is part of the key
# so a Base call and a Solana call don't collide.

_sync_clients: Dict[str, Any] = {}    # may be LLMClient or SolanaLLMClient
_async_clients: Dict[str, AsyncLLMClient] = {}
_image_clients: Dict[str, ImageClient] = {}
_lock = threading.Lock()


def _wallet_env_var(api_url: Optional[str]) -> str:
    """Which env var to consult for the default wallet on this chain."""
    return "SOLANA_WALLET_KEY" if _is_solana_url(api_url) else "BLOCKRUN_WALLET_KEY"


def _client_key(api_url: Optional[str], private_key: Optional[str]) -> str:
    chain = "solana" if _is_solana_url(api_url) else "base"
    fallback_env = os.environ.get(_wallet_env_var(api_url), "")
    return f"{chain}::{api_url or ''}::{private_key or fallback_env}"


def get_sync_client(
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Union[LLMClient, "SolanaLLMClient"]:  # type: ignore[name-defined]
    """Return a cached sync client for the given creds/url.

    Routes to :class:`SolanaLLMClient` when ``api_url`` points at
    ``sol.blockrun.ai``, otherwise :class:`LLMClient` (Base).
    """
    key = _client_key(api_url, private_key)
    with _lock:
        client = _sync_clients.get(key)
        if client is None:
            if _is_solana_url(api_url):
                if not _HAS_SOLANA:
                    raise ImportError(
                        "Solana support requires the solana extra. "
                        "Install with: pip install 'blockrun-llm[solana]'"
                    )
                # SolanaLLMClient also reads from SOLANA_WALLET_KEY if no
                # explicit key was passed.
                client = SolanaLLMClient(
                    private_key=private_key,
                    api_url=api_url or SOLANA_API_URL,
                )
            else:
                client = LLMClient(private_key=private_key, api_url=api_url)
            _sync_clients[key] = client
        return client


def get_async_client(
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Union[AsyncLLMClient, "AsyncSolanaLLMClient"]:  # type: ignore[name-defined]
    """Return a cached async client for the given creds/url.

    Routes to :class:`AsyncSolanaLLMClient` when ``api_url`` points at
    ``sol.blockrun.ai``, otherwise :class:`AsyncLLMClient` (Base).
    Requires ``blockrun-llm>=0.22.0`` for the async Solana client.
    """
    is_solana = _is_solana_url(api_url)
    key = _client_key(api_url, private_key)
    with _lock:
        client = _async_clients.get(key)
        if client is None:
            if is_solana:
                if not _HAS_SOLANA or AsyncSolanaLLMClient is None:
                    raise ImportError(
                        "Solana support requires the solana extra. "
                        "Install with: pip install 'blockrun-litellm[solana]'"
                    )
                client = AsyncSolanaLLMClient(
                    private_key=private_key,
                    api_url=api_url or SOLANA_API_URL,
                )
            else:
                client = AsyncLLMClient(private_key=private_key, api_url=api_url)
            _async_clients[key] = client
        return client


# ---------------------------------------------------------------------------
# Request normalization
# ---------------------------------------------------------------------------

# OpenAI-style params that blockrun-llm's chat methods accept directly.
# Anything outside this set is dropped — LiteLLM tends to forward
# provider-specific kwargs that don't apply here.
#
# Since blockrun-llm 0.22.1 (Solana) / 0.20.0 (Base), the set is the same
# on both chains — function calling (``tools`` / ``tool_choice``) works on
# either path because the BlockRun gateway forwards them to the upstream
# model unchanged; the chain only differs in the payment leg.
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


def _filter_kwargs(payload: Dict[str, Any], *, is_solana: bool = False) -> Dict[str, Any]:
    """Whitelist OpenAI-format kwargs into the shape blockrun-llm wants.

    The ``is_solana`` parameter is currently unused — kept in the signature
    for backwards compatibility / future chain-specific filtering. Both
    chains accept the same set of kwargs.

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

    Routes to Solana (``SolanaLLMClient``) when ``api_url`` points at
    ``sol.blockrun.ai``, otherwise Base (``LLMClient``). Any LiteLLM
    ``blockrun/`` prefix should already be stripped by the caller.

    For ``stream=True``, use :func:`chat_completion_stream_sync` instead.
    """
    openai_kwargs.pop("stream", None)
    is_solana = _is_solana_url(api_url)
    kwargs = _filter_kwargs(openai_kwargs, is_solana=is_solana)
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
    """Async variant of :func:`chat_completion_sync`.

    **Base only today.** Solana ``api_url`` raises ``NotImplementedError``
    via :func:`get_async_client` since the SDK has no async Solana client.
    """
    openai_kwargs.pop("stream", None)
    is_solana = _is_solana_url(api_url)
    kwargs = _filter_kwargs(openai_kwargs, is_solana=is_solana)
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
    Stream a chat completion via the SDK's ``chat_completion_stream``.

    Routes to Solana or Base based on ``api_url``. Yields
    :class:`ChatCompletionChunk` objects (OpenAI chunk schema). Caller
    is responsible for downstream formatting (LiteLLM
    ``GenericStreamingChunk``, FastAPI ``data: <json>\\n\\n``, etc.).
    """
    openai_kwargs.pop("stream", None)
    is_solana = _is_solana_url(api_url)
    kwargs = _filter_kwargs(openai_kwargs, is_solana=is_solana)
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
    """Async variant of :func:`chat_completion_stream_sync`.

    **Base only today.** A Solana ``api_url`` raises
    ``NotImplementedError`` since the SDK has no async Solana client.
    """
    openai_kwargs.pop("stream", None)
    is_solana = _is_solana_url(api_url)
    kwargs = _filter_kwargs(openai_kwargs, is_solana=is_solana)
    client = get_async_client(api_url=api_url, private_key=private_key)
    async for chunk in client.chat_completion_stream(
        model=model, messages=messages, **kwargs
    ):
        yield chunk


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
# ImageClient is sync-only in blockrun-llm; async callers run it in a thread.


def get_image_client(
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> ImageClient:
    key = _client_key(api_url, private_key)
    with _lock:
        client = _image_clients.get(key)
        if client is None:
            client = ImageClient(private_key=private_key, api_url=api_url)
            _image_clients[key] = client
        return client


def image_generation_sync(
    prompt: str,
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Dict[str, Any]:
    client = get_image_client(api_url=api_url, private_key=private_key)
    response = client.generate(prompt, model=model, size=size, n=n)
    return response.model_dump(exclude_none=True)


async def image_generation_async(
    prompt: str,
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Dict[str, Any]:
    client = get_image_client(api_url=api_url, private_key=private_key)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: client.generate(prompt, model=model, size=size, n=n)
    )
    return response.model_dump(exclude_none=True)


__all__ = [
    "chat_completion_sync",
    "chat_completion_async",
    "chat_completion_stream_sync",
    "chat_completion_stream_async",
    "get_sync_client",
    "get_async_client",
    "get_image_client",
    "image_generation_sync",
    "image_generation_async",
    "APIError",
    "PaymentError",
]
