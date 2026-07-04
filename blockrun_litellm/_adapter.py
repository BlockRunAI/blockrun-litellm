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
import concurrent.futures
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

# Default chat HTTP timeout (seconds) for the SDK clients the adapter builds.
# The SDK default was 120s — too low for reasoning models (opus-4.8 /
# deepseek-v4-pro routinely take 200–300s). Passed explicitly so the adapter
# doesn't depend on the installed SDK version's default. Override via
# BLOCKRUN_CHAT_TIMEOUT. NB: for streaming this is a per-chunk read timeout;
# for non-stream it's the whole-call timeout.
_DEFAULT_CHAT_TIMEOUT_S = 600.0


def _chat_timeout() -> float:
    """Resolve the chat timeout, falling back on a malformed env var.

    Mirrors :func:`_solana_image_timeout` — a non-numeric BLOCKRUN_CHAT_TIMEOUT
    (e.g. ``"600s"``) must NOT crash module import, which would take down both
    the provider and the proxy that import this module.
    """
    raw = os.environ.get("BLOCKRUN_CHAT_TIMEOUT")
    if not raw:
        return _DEFAULT_CHAT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_CHAT_TIMEOUT_S


_CHAT_TIMEOUT = _chat_timeout()

_sync_clients: Dict[str, Any] = {}    # may be LLMClient or SolanaLLMClient
_async_clients: Dict[str, AsyncLLMClient] = {}
_image_clients: Dict[str, Any] = {}  # ImageClient (Base) or SolanaLLMClient (Solana)
_lock = threading.Lock()

# Bounded thread pool for image generation (ImageClient is sync-only).
# Capped at 20 so that high-concurrency image requests don't spawn unlimited
# threads and exhaust memory. Matches the default BLOCKRUN_MAX_CONCURRENT.
_image_executor: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(max_workers=20)
)


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
                    timeout=_CHAT_TIMEOUT,
                )
            else:
                client = LLMClient(
                    private_key=private_key, api_url=api_url, timeout=_CHAT_TIMEOUT
                )
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
                    timeout=_CHAT_TIMEOUT,
                )
            else:
                client = AsyncLLMClient(
                    private_key=private_key, api_url=api_url, timeout=_CHAT_TIMEOUT
                )
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
    # Reasoning controls — the gateway forwards these to the upstream model
    # (e.g. Anthropic extended thinking). Without them in the whitelist they
    # were silently dropped, so callers could never trigger thinking via litellm.
    "reasoning_effort",
    "thinking",
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
# Real-cost extraction
# ---------------------------------------------------------------------------
# LiteLLM bills off a token-count × list-price estimate, which does NOT match
# BlockRun's real x402 charge (the gateway price carries a per-call floor +
# margin). We surface the SDK's real charge so callers can report the actual
# wallet deduction instead. The authoritative source is the per-call value the
# SDK attaches to the response (``response.cost_usd``, since blockrun-llm 1.3);
# ``client._last_call_cost`` is a best-effort fallback for older SDKs (note it
# goes stale on free/cached calls and is racy under shared-client concurrency).
_BLOCKRUN_META_KEY = "_blockrun"


def _strip_real_cost(payload: Dict[str, Any], client: Any) -> Dict[str, Any]:
    """Pop the SDK-attached cost/settlement out of the dumped payload and return
    a ``{cost_usd, settlement}`` meta dict (cost may be ``None`` if unavailable)."""
    cost = payload.pop("cost_usd", None)
    settlement = payload.pop("settlement", None)
    if cost is None:
        cost = getattr(client, "_last_call_cost", None)
    return {"cost_usd": cost, "settlement": settlement}


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
    payload = response.model_dump(exclude_none=True)
    payload[_BLOCKRUN_META_KEY] = _strip_real_cost(payload, client)
    return payload


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
    payload = response.model_dump(exclude_none=True)
    payload[_BLOCKRUN_META_KEY] = _strip_real_cost(payload, client)
    return payload


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


# Per-image-request timeout (seconds) for the Solana image client. Default 300s
# leaves headroom above the SDK's 200s ``image_timeout`` for the slow tail of
# ``openai/gpt-image-2`` (public reports cite 145-280s). Read at call time so
# ``BLOCKRUN_SOLANA_IMAGE_TIMEOUT`` can be tuned without a process restart.
_DEFAULT_SOLANA_IMAGE_TIMEOUT_S = 300.0


def _solana_image_timeout() -> float:
    raw = os.environ.get("BLOCKRUN_SOLANA_IMAGE_TIMEOUT")
    if not raw:
        return _DEFAULT_SOLANA_IMAGE_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_SOLANA_IMAGE_TIMEOUT_S


def get_image_client(
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Union[ImageClient, "SolanaLLMClient"]:  # type: ignore[name-defined]
    """Return a cached image client for the given creds/url.

    Routes to :class:`SolanaLLMClient` when ``api_url`` points at
    ``sol.blockrun.ai``, otherwise :class:`ImageClient` (Base).

    The Solana branch is required because ``ImageClient`` only signs EIP-712
    over the EVM ``Account`` — sending those payments to the Solana gateway
    fails at x402 settlement (``transaction_simulation_failed``).
    ``SolanaLLMClient`` exposes ``.image()`` / ``.image_edit()`` that hit the
    same ``/v1/images/*`` endpoints with SVM-scheme x402 payments.
    """
    key = _client_key(api_url, private_key)
    with _lock:
        client = _image_clients.get(key)
        if client is None:
            if _is_solana_url(api_url):
                if not _HAS_SOLANA or SolanaLLMClient is None:
                    raise ImportError(
                        "Solana support requires the solana extra. "
                        "Install with: pip install 'blockrun-litellm[solana]'"
                    )
                client = SolanaLLMClient(
                    private_key=private_key,
                    api_url=api_url or SOLANA_API_URL,
                    # Raise the per-image-request timeout ceiling. The SDK caps
                    # each image POST at ``image_timeout`` (SolanaLLMClient
                    # default 200s); slow models such as ``openai/gpt-image-2``
                    # can exceed that on the synchronous Solana path, so the
                    # sidecar would otherwise throw ``httpx.ReadTimeout`` mid-
                    # generation. NOTE: the general ``timeout=`` kwarg is the
                    # chat baseline and is overridden per-request for images
                    # (``_request_image_with_payment`` passes ``image_timeout``),
                    # so ``image_timeout=`` is the knob that actually governs
                    # image calls. Tunable via BLOCKRUN_SOLANA_IMAGE_TIMEOUT
                    # for ops without a redeploy.
                    image_timeout=_solana_image_timeout(),
                )
            else:
                client = ImageClient(private_key=private_key, api_url=api_url)
            _image_clients[key] = client
        return client


def _invoke_image_generate(client: Any, prompt: str, *, model, size, n):
    """Dispatch ``generate`` (Base ImageClient) vs ``image`` (SolanaLLMClient).

    ``ImageClient.generate`` and ``SolanaLLMClient.image`` are intentionally
    named differently in the SDK but accept the same call shape.
    """
    if _HAS_SOLANA and SolanaLLMClient is not None and isinstance(client, SolanaLLMClient):
        # SolanaLLMClient.image requires non-None model/size (no class-level defaults).
        kwargs: Dict[str, Any] = {"n": n}
        if model is not None:
            kwargs["model"] = model
        if size is not None:
            kwargs["size"] = size
        return client.image(prompt, **kwargs)
    return client.generate(prompt, model=model, size=size, n=n)


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
    response = _invoke_image_generate(client, prompt, model=model, size=size, n=n)
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
        _image_executor,
        lambda: _invoke_image_generate(client, prompt, model=model, size=size, n=n),
    )
    return response.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Video / music / speech generation (Base dedicated clients vs Solana unified)
# ---------------------------------------------------------------------------
# On Base each medium has its own SDK client (VideoClient/MusicClient/
# SpeechClient). On Solana every medium is a method on the one SolanaLLMClient
# (which get_image_client already builds + caches). All of these clients are
# sync-only, so async callers run them in the shared _image_executor thread
# pool — same pattern as image_generation_async.

_media_clients: Dict[str, Any] = {}


def _get_base_media_client(base_cls: Any, api_url: Optional[str], private_key: Optional[str]) -> Any:
    """Cache + return a Base dedicated media client (VideoClient/MusicClient/…)."""
    key = f"{base_cls.__name__}::{_client_key(api_url, private_key)}"
    with _lock:
        client = _media_clients.get(key)
        if client is None:
            client = base_cls(private_key=private_key, api_url=api_url)
            _media_clients[key] = client
        return client


def _is_solana_client(client: Any) -> bool:
    return _HAS_SOLANA and SolanaLLMClient is not None and isinstance(client, SolanaLLMClient)


def get_video_client(api_url: Optional[str] = None, private_key: Optional[str] = None) -> Any:
    """VideoClient (Base) or the unified SolanaLLMClient (Solana)."""
    if _is_solana_url(api_url):
        return get_image_client(api_url=api_url, private_key=private_key)
    from blockrun_llm import VideoClient

    return _get_base_media_client(VideoClient, api_url, private_key)


def get_music_client(api_url: Optional[str] = None, private_key: Optional[str] = None) -> Any:
    """MusicClient (Base) or the unified SolanaLLMClient (Solana)."""
    if _is_solana_url(api_url):
        return get_image_client(api_url=api_url, private_key=private_key)
    from blockrun_llm import MusicClient

    return _get_base_media_client(MusicClient, api_url, private_key)


def get_speech_client(api_url: Optional[str] = None, private_key: Optional[str] = None) -> Any:
    """SpeechClient (Base) or the unified SolanaLLMClient (Solana). Serves both
    TTS (speech) and sound-effects."""
    if _is_solana_url(api_url):
        return get_image_client(api_url=api_url, private_key=private_key)
    from blockrun_llm import SpeechClient

    return _get_base_media_client(SpeechClient, api_url, private_key)


def _run_media(func: Any) -> Any:
    """Run a sync SDK media call in the shared executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_image_executor, func)


async def video_generation_async(
    prompt: str,
    *,
    model: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    **params: Any,
) -> Dict[str, Any]:
    """Generate a video. Extra kwargs (image_url, duration_seconds, resolution,
    aspect_ratio, generate_audio, seed, reference_image_urls, real_face_asset_id,
    last_frame_url, watermark, return_last_frame, budget_seconds) forward to the
    SDK. ``timeout`` is only honored on Solana (Base VideoClient has no such arg)."""
    client = get_video_client(api_url=api_url, private_key=private_key)
    params = {k: v for k, v in params.items() if v is not None}
    if _is_solana_client(client):
        response = await _run_media(lambda: client.video(prompt, model=model, **params))
    else:
        params.pop("timeout", None)  # Base VideoClient.generate has no timeout kwarg
        response = await _run_media(lambda: client.generate(prompt, model=model, **params))
    return response.model_dump(exclude_none=True)


async def music_generation_async(
    prompt: str,
    *,
    model: Optional[str] = None,
    instrumental: bool = True,
    lyrics: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a music track."""
    client = get_music_client(api_url=api_url, private_key=private_key)
    call = (
        (lambda: client.music(prompt, model=model, instrumental=instrumental, lyrics=lyrics))
        if _is_solana_client(client)
        else (
            lambda: client.generate(prompt, model=model, instrumental=instrumental, lyrics=lyrics)
        )
    )
    response = await _run_media(call)
    return response.model_dump(exclude_none=True)


async def speech_generation_async(
    input: str,
    *,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize speech (TTS)."""
    client = get_speech_client(api_url=api_url, private_key=private_key)
    kw = {"model": model, "voice": voice, "response_format": response_format, "speed": speed}
    kw = {k: v for k, v in kw.items() if v is not None}
    call = (
        (lambda: client.speech(input, **kw))
        if _is_solana_client(client)
        else (lambda: client.generate(input, **kw))
    )
    response = await _run_media(call)
    return response.model_dump(exclude_none=True)


async def sound_effect_async(
    text: str,
    *,
    model: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    prompt_influence: Optional[float] = None,
    response_format: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a cinematic sound effect."""
    client = get_speech_client(api_url=api_url, private_key=private_key)
    kw = {
        "model": model,
        "duration_seconds": duration_seconds,
        "prompt_influence": prompt_influence,
        "response_format": response_format,
    }
    kw = {k: v for k, v in kw.items() if v is not None}
    # Both SolanaLLMClient and SpeechClient expose .sound_effect with the same shape.
    response = await _run_media(lambda: client.sound_effect(text, **kw))
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
    "get_video_client",
    "get_music_client",
    "get_speech_client",
    "video_generation_async",
    "music_generation_async",
    "speech_generation_async",
    "sound_effect_async",
    "APIError",
    "PaymentError",
]
