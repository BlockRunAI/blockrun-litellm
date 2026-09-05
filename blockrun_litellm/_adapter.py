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
import logging
import os
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from blockrun_llm import AsyncLLMClient, ImageClient, LLMClient
from blockrun_llm.types import APIError, ChatCompletionChunk, PaymentError

from blockrun_litellm import _apikey

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

_log = logging.getLogger(__name__)


def _canonical_video_model(model: Optional[str]) -> Optional[str]:
    """Namespace the short video model ids accepted by OpenAI-style clients.

    Token360 documents ``seedance-2.0-fast`` while BlockRun's gateway catalog
    uses canonical provider-qualified ids.  The proxy must bridge that naming
    difference before calling the SDK's flat ``/videos/generations`` surface.
    The gateway does the same rewrite itself, but only in its ``content[]``
    handler for ``POST /v1/videos`` — never on the flat surface the SDK posts
    to, and the SDK doesn't normalize either.

    The id is lowercased on the way through: the catalog lookup is an exact
    ``==`` match, so a case-preserved ``bytedance/Seedance-2.0-Fast`` would 400
    just like the bare id it replaced.

    Only unambiguous bare ids are mapped, and only ones whose vendor is implied
    by the name itself.  ``sora-2`` is deliberately absent: the catalog ships it
    under *two* vendors (``azure/sora-2``, available; ``openai/sora-2``, not),
    so there is no namespace to infer — guessing would silently pick a vendor on
    the caller's behalf, and the choice would flip meaning the day availability
    does.  An unmapped bare id reaches the gateway untouched and 400s with the
    full list of real ids, which is the better answer than a confident guess.
    """
    if not isinstance(model, str):
        return model
    bare = model.strip().lower()
    if bare.startswith("seedance-"):
        return f"bytedance/{bare}"
    if bare == "grok-imagine-video":
        return f"xai/{bare}"
    return model


# ---------------------------------------------------------------------------
# Chain selection (wallet rail only)
# ---------------------------------------------------------------------------
# Solana is the default chain as of 0.10.0. It settles in roughly a second for
# a fraction of a cent, where a Base settlement is both slower and dearer, so
# it is the chain a new caller should land on without having to know there was
# a choice.
#
# Flipping a default cannot brick a running deployment, though, and a host that
# has only ever held a Base wallet would otherwise start failing the moment it
# upgraded — its key is hex, and the SVM signer cannot use it. So the flip is
# conditional on what is actually on the box: an implicit default picks Base,
# loudly, when Base is the only credential present. Anything explicit
# (``BLOCKRUN_CHAIN``, ``BLOCKRUN_API_URL``, an ``api_url`` argument) always
# wins over both, including a ``BLOCKRUN_CHAIN=solana`` on a host with no
# Solana wallet — that must fail with "no wallet", not silently serve Base.

_CHAIN_ALIASES: Dict[str, str] = {
    "solana": SOLANA_API_URL,
    "sol": SOLANA_API_URL,
    "svm": SOLANA_API_URL,
    "base": BASE_API_URL,
    "evm": BASE_API_URL,
}

_SOLANA_KEY_ENVS = ("SOLANA_WALLET_KEY",)
_BASE_KEY_ENVS = ("BLOCKRUN_WALLET_KEY", "BASE_CHAIN_WALLET_KEY")
# Written by the SDK's interactive wallet setup; the same files it auto-loads.
_SOLANA_SESSION = Path.home() / ".blockrun" / ".solana-session"
_BASE_SESSION = Path.home() / ".blockrun" / ".session"


def _has_wallet(envs: Any, session: Path) -> bool:
    if any(os.environ.get(name) for name in envs):
        return True
    try:
        return session.is_file()
    except OSError:  # unreadable home dir — treat as absent, never crash
        return False


# The credential probe touches the filesystem, so it is memoized against the
# env vars that can change its answer. Re-statting ~/.blockrun on every request
# would be a syscall per completion for a value that almost never moves.
_default_url_cache: Dict[Any, str] = {}


def _default_wallet_api_url() -> str:
    """Gateway URL when nothing explicit says which chain to use."""
    chain = (os.environ.get("BLOCKRUN_CHAIN") or "").strip().lower()
    if chain:
        resolved = _CHAIN_ALIASES.get(chain)
        if resolved:
            return resolved
        _log.warning(
            "BLOCKRUN_CHAIN=%r is not a known chain (expected one of %s); "
            "falling back to auto-detection",
            chain,
            ", ".join(sorted(_CHAIN_ALIASES)),
        )
    cache_key = tuple(os.environ.get(name, "") for name in _SOLANA_KEY_ENVS + _BASE_KEY_ENVS)
    cached = _default_url_cache.get(cache_key)
    if cached is not None:
        return cached
    if not _has_wallet(_SOLANA_KEY_ENVS, _SOLANA_SESSION) and _has_wallet(
        _BASE_KEY_ENVS, _BASE_SESSION
    ):
        _log.warning(
            "Solana is now the default BlockRun chain, but only a Base wallet was "
            "found on this host, so this process keeps using Base. Set "
            "BLOCKRUN_CHAIN=base to make that explicit (and silence this warning), "
            "or set SOLANA_WALLET_KEY to move to Solana."
        )
        resolved = BASE_API_URL
    else:
        resolved = SOLANA_API_URL
    _default_url_cache[cache_key] = resolved
    return resolved


def resolve_api_url(api_url: Optional[str] = None) -> str:
    """The gateway URL a wallet-rail call will actually use.

    Precedence: explicit argument, then ``BLOCKRUN_API_URL``, then the chain
    default. Always concrete — callers pass the result straight to an SDK
    client rather than relying on the SDK's own (Base) default, which no longer
    matches ours.
    """
    return api_url or os.environ.get("BLOCKRUN_API_URL") or _default_wallet_api_url()


def _reset_chain_cache_for_tests() -> None:
    _default_url_cache.clear()


def _is_solana_url(api_url: Optional[str]) -> bool:
    """Sniff whether the effective gateway URL points at Solana.

    Falls back to ``BLOCKRUN_API_URL`` and then the chain default when no
    explicit ``api_url`` is passed. This matters for the FastAPI sidecar: the
    request handlers don't forward an ``api_url`` arg, so without the
    env-var fallback we'd silently route Solana traffic to the Base
    async client and crash inside the EVM payment encoder
    (``eth_abi.AddressEncoder`` rejects base58 mint addresses).
    """
    return "sol.blockrun.ai" in resolve_api_url(api_url)


# ---------------------------------------------------------------------------
# Rail selection (API key vs wallet)
# ---------------------------------------------------------------------------


def _route_key(api_key: Optional[str], private_key: Optional[str]) -> Optional[str]:
    """The BlockRun API key serving this call, or ``None`` for the wallet rail.

    ``private_key`` is consulted too because that is the argument LiteLLM's
    ``api_key`` lands in (see :mod:`blockrun_litellm.provider`), so
    ``litellm.completion(..., api_key="brk_live_...")`` picks the account rail
    without the caller needing a second parameter name. A hex or base58 wallet
    key cannot be mistaken for one — see :func:`_apikey.looks_like_api_key`.
    """
    return _apikey.resolve_api_key(api_key, private_key)


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

_sync_clients: Dict[str, Any] = {}  # may be LLMClient or SolanaLLMClient
_async_clients: Dict[str, AsyncLLMClient] = {}
_image_clients: Dict[str, Any] = {}  # ImageClient (Base) or SolanaLLMClient (Solana)
_lock = threading.Lock()

# Bounded thread pool for image generation (ImageClient is sync-only).
# Capped at 20 so that high-concurrency image requests don't spawn unlimited
# threads and exhaust memory. Matches the default BLOCKRUN_MAX_CONCURRENT.
_image_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=20
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
                    api_url=resolve_api_url(api_url),
                    timeout=_CHAT_TIMEOUT,
                )
            else:
                client = LLMClient(
                    private_key=private_key,
                    api_url=resolve_api_url(api_url),
                    timeout=_CHAT_TIMEOUT,
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
                    api_url=resolve_api_url(api_url),
                    timeout=_CHAT_TIMEOUT,
                )
            else:
                client = AsyncLLMClient(
                    private_key=private_key,
                    api_url=resolve_api_url(api_url),
                    timeout=_CHAT_TIMEOUT,
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


# The API-key rail settles against prepaid credit, off chain, so there is no
# per-call charge to report and no settlement receipt to decode. Saying so
# explicitly — rather than leaving the key absent — keeps the downstream
# "real cost or estimate?" branch a single lookup on both rails, and makes the
# absence deliberate rather than a hole someone later fills with an estimate.
_NO_ONCHAIN_COST: Dict[str, Any] = {"cost_usd": None, "settlement": None, "rail": "api_key"}


# ---------------------------------------------------------------------------
# Non-streaming entrypoints
# ---------------------------------------------------------------------------


def chat_completion_sync(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
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
    key = _route_key(api_key, private_key)
    if key:
        payload = _apikey.post_json(
            "/v1/chat/completions",
            {"model": model, "messages": messages, **_filter_kwargs(openai_kwargs)},
            api_key=key,
            timeout=_CHAT_TIMEOUT,
        )
        payload[_BLOCKRUN_META_KEY] = dict(_NO_ONCHAIN_COST)
        return payload
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
    api_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> Dict[str, Any]:
    """Async variant of :func:`chat_completion_sync`.

    **Base only today** on the wallet rail: a Solana ``api_url`` raises
    ``NotImplementedError`` via :func:`get_async_client` since the SDK has no
    async Solana client. The API-key rail has no such limit — it is plain HTTP
    with no chain-specific signer — so async works there regardless.
    """
    openai_kwargs.pop("stream", None)
    key = _route_key(api_key, private_key)
    if key:
        payload = await _apikey.apost_json(
            "/v1/chat/completions",
            {"model": model, "messages": messages, **_filter_kwargs(openai_kwargs)},
            api_key=key,
            timeout=_CHAT_TIMEOUT,
        )
        payload[_BLOCKRUN_META_KEY] = dict(_NO_ONCHAIN_COST)
        return payload
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
    api_key: Optional[str] = None,
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
    key = _route_key(api_key, private_key)
    if key:
        yield from _apikey.stream_chat(
            {"model": model, "messages": messages, **_filter_kwargs(openai_kwargs)},
            api_key=key,
            timeout=_CHAT_TIMEOUT,
        )
        return
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
    api_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> AsyncIterator[ChatCompletionChunk]:
    """Async variant of :func:`chat_completion_stream_sync`.

    **Base only today** on the wallet rail: a Solana ``api_url`` raises
    ``NotImplementedError`` since the SDK has no async Solana client. The
    API-key rail is plain HTTP and has no such limit.
    """
    openai_kwargs.pop("stream", None)
    key = _route_key(api_key, private_key)
    if key:
        async for chunk in _apikey.astream_chat(
            {"model": model, "messages": messages, **_filter_kwargs(openai_kwargs)},
            api_key=key,
            timeout=_CHAT_TIMEOUT,
        ):
            yield chunk
        return
    is_solana = _is_solana_url(api_url)
    kwargs = _filter_kwargs(openai_kwargs, is_solana=is_solana)
    client = get_async_client(api_url=api_url, private_key=private_key)
    async for chunk in client.chat_completion_stream(model=model, messages=messages, **kwargs):
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
                    api_url=resolve_api_url(api_url),
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
                client = ImageClient(private_key=private_key, api_url=resolve_api_url(api_url))
            _image_clients[key] = client
        return client


def _is_solana_image_client(client: Any) -> bool:
    return _HAS_SOLANA and SolanaLLMClient is not None and isinstance(client, SolanaLLMClient)


# `quality` exists only on the Solana image surface: the Base gateway defines no
# such field and strips unknown keys, and the SDK's ImageClient raises TypeError
# on it rather than forward it.
#
# 0.7.0 refused it on Base with a 400. That was wrong: 0.6.1 never read `quality`
# at all, so callers using it got a 200 — and `quality` is a first-class OpenAI
# Images parameter on a route documented as DALL-E compatible. Turning a
# previously-working, spec-compliant request into a hard failure is a breaking
# change, not something to slip into a minor bump as a side effect of adding
# Solana support.
#
# So: drop it on Base, as 0.6.1 effectively did, but say so — a warning log plus
# an `x-blockrun-warning` response header, so the value doesn't vanish in silence
# the way it used to.
_QUALITY_UNSUPPORTED_ON_BASE = (
    "`quality` is supported on Solana only (openai/gpt-image-* via "
    "sol.blockrun.ai) and was ignored: the Base gateway has no quality field. "
    "Point BLOCKRUN_API_URL at the Solana gateway to use it."
)


def _invoke_image_generate(client: Any, prompt: str, *, model, size, n, quality=None):
    """Dispatch ``generate`` (Base ImageClient) vs ``image`` (SolanaLLMClient).

    ``ImageClient.generate`` and ``SolanaLLMClient.image`` are intentionally
    named differently in the SDK but accept the same call shape, except for
    ``quality`` — Solana only, see :data:`_QUALITY_UNSUPPORTED_ON_BASE`.
    """
    # Omit optional values instead of passing ``None``. This keeps the adapter
    # compatible with SDK releases from before an optional parameter was added.
    kwargs: Dict[str, Any] = {"n": n}
    if model is not None:
        kwargs["model"] = model
    if size is not None:
        kwargs["size"] = size
    if _is_solana_image_client(client):
        if quality is not None:
            kwargs["quality"] = quality
        # SolanaLLMClient.image requires non-None model/size (no class-level defaults).
        return client.image(prompt, **kwargs)
    if quality is not None:
        _log.warning(_QUALITY_UNSUPPORTED_ON_BASE)
    return client.generate(prompt, **kwargs)


def _invoke_image_edit(
    client: Any,
    prompt: str,
    image: Any,
    *,
    model: Optional[str],
    mask: Optional[str],
    size: Optional[str],
    n: int,
    quality: Optional[str] = None,
):
    kwargs: Dict[str, Any] = {"n": n}
    for key, value in {
        "model": model,
        "mask": mask,
        "size": size,
    }.items():
        if value is not None:
            kwargs[key] = value
    if _is_solana_image_client(client):
        if quality is not None:
            kwargs["quality"] = quality
        return client.image_edit(prompt, image, **kwargs)
    if quality is not None:
        _log.warning(_QUALITY_UNSUPPORTED_ON_BASE)
    return client.edit(prompt, image, **kwargs)


# ---------------------------------------------------------------------------
# API-key bodies for the media surfaces
# ---------------------------------------------------------------------------
# The SDK builds these bodies itself and they are not exported, so the API-key
# rail rebuilds them here. They must match what the SDK sends — including its
# defaults for an omitted model or size — or the two rails would answer the
# same call with different pictures. The defaults are read off the SDK classes
# rather than copied, so a default that moves upstream moves here too.


def _image_defaults() -> tuple:
    return ImageClient.DEFAULT_MODEL, ImageClient.DEFAULT_SIZE


def _image_body(
    prompt: str,
    *,
    model: Optional[str],
    size: Optional[str],
    n: int,
    quality: Optional[str],
) -> Dict[str, Any]:
    default_model, default_size = _image_defaults()
    body: Dict[str, Any] = {
        "model": model or default_model,
        "prompt": prompt,
        "size": size or default_size,
        "n": n,
    }
    # No chain here, so no `quality` suppression: the Solana-only rule exists
    # because the Base gateway has no such field, and which gateway serves an
    # account call is BlockRun's routing decision, not the caller's. Forwarding
    # it lets the account API accept or ignore it the way it does for any other
    # OpenAI Images parameter.
    if quality is not None:
        body["quality"] = quality
    return body


def _image_edit_body(
    prompt: str,
    image: Any,
    *,
    model: Optional[str],
    mask: Optional[str],
    size: Optional[str],
    n: int,
    quality: Optional[str],
) -> Dict[str, Any]:
    # ``openai/gpt-image-2`` is ImageClient.edit's own hardcoded default and has
    # no class constant to read, so it is repeated rather than referenced.
    _, default_size = _image_defaults()
    body: Dict[str, Any] = {
        "model": model or "openai/gpt-image-2",
        "prompt": prompt,
        "image": image,
        "size": size or default_size,
        "n": n,
    }
    if mask is not None:
        body["mask"] = mask
    if quality is not None:
        body["quality"] = quality
    return body


def image_generation_sync(
    prompt: str,
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    quality: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    key = _route_key(api_key, private_key)
    if key:
        return _apikey.post_json(
            "/v1/images/generations",
            _image_body(prompt, model=model, size=size, n=n, quality=quality),
            api_key=key,
            timeout=_solana_image_timeout(),
        )
    client = get_image_client(api_url=api_url, private_key=private_key)
    response = _invoke_image_generate(
        client, prompt, model=model, size=size, n=n, quality=quality
    )
    return response.model_dump(exclude_none=True)


async def image_generation_async(
    prompt: str,
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    quality: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    key = _route_key(api_key, private_key)
    if key:
        return await _apikey.apost_json(
            "/v1/images/generations",
            _image_body(prompt, model=model, size=size, n=n, quality=quality),
            api_key=key,
            timeout=_solana_image_timeout(),
        )
    client = get_image_client(api_url=api_url, private_key=private_key)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        _image_executor,
        lambda: _invoke_image_generate(
            client, prompt, model=model, size=size, n=n, quality=quality
        ),
    )
    return response.model_dump(exclude_none=True)


def image_edit_sync(
    prompt: str,
    image: Any,
    *,
    model: Optional[str] = None,
    mask: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    quality: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    key = _route_key(api_key, private_key)
    if key:
        return _apikey.post_json(
            "/v1/images/edits",
            _image_edit_body(
                prompt, image, model=model, mask=mask, size=size, n=n, quality=quality
            ),
            api_key=key,
            timeout=_solana_image_timeout(),
        )
    client = get_image_client(api_url=api_url, private_key=private_key)
    response = _invoke_image_edit(
        client,
        prompt,
        image,
        model=model,
        mask=mask,
        size=size,
        n=n,
        quality=quality,
    )
    return response.model_dump(exclude_none=True)


async def image_edit_async(
    prompt: str,
    image: Any,
    *,
    model: Optional[str] = None,
    mask: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1,
    quality: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    key = _route_key(api_key, private_key)
    if key:
        return await _apikey.apost_json(
            "/v1/images/edits",
            _image_edit_body(
                prompt, image, model=model, mask=mask, size=size, n=n, quality=quality
            ),
            api_key=key,
            timeout=_solana_image_timeout(),
        )
    client = get_image_client(api_url=api_url, private_key=private_key)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        _image_executor,
        lambda: _invoke_image_edit(
            client,
            prompt,
            image,
            model=model,
            mask=mask,
            size=size,
            n=n,
            quality=quality,
        ),
    )
    return response.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Video / music / speech generation (Base dedicated clients vs Solana unified)
# ---------------------------------------------------------------------------
# On Base each medium has its own SDK client (VideoClient/MusicClient/
# SpeechClient). On Solana every medium is a method on the one SolanaLLMClient
# (which get_image_client already builds + caches). All of these clients are
# sync-only, so async callers run them in a thread pool. Fast media (speech,
# sound-effects) share _image_executor with images; long-running media (video
# 60-900s, music 60-210s) get their own smaller pool so a burst of video jobs
# can't pin all 20 image/speech threads for 15 minutes.

_media_clients: Dict[str, Any] = {}

_LONG_MEDIA_THREADS: int = int(os.environ.get("BLOCKRUN_LONG_MEDIA_THREADS", "8"))
_long_media_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_LONG_MEDIA_THREADS
)

# Server-side wall-clock ceilings. budget_seconds/timeout arrive from the
# request body, so without a clamp a caller could pin a worker thread for a
# day. The video cap matches the SDK's DEFAULT_GENERATE_BUDGET_SECONDS; the
# await ceilings add margin over the longest legitimate SDK call so the
# coroutine (and its semaphore permit) is always released even if the SDK
# thread wedges. NB: wait_for can't kill the worker thread — it only frees
# the awaiting coroutine; the orphaned thread exits when the SDK call does.
_VIDEO_BUDGET_CAP_S = 900.0
_VIDEO_AWAIT_CEILING_S = _VIDEO_BUDGET_CAP_S + 120.0
_MUSIC_AWAIT_CEILING_S = 300.0  # MusicClient DEFAULT_TIMEOUT 210s + margin
_SPEECH_AWAIT_CEILING_S = 150.0  # SpeechClient DEFAULT_TIMEOUT 120s + margin

# Lazy imports keep module import working on SDK versions predating a client.
_BASE_MEDIA_CLASSES = {"video": "VideoClient", "music": "MusicClient", "speech": "SpeechClient"}


def _sdk_default(cls_name: str, attr: str) -> Optional[str]:
    """A default model the SDK would have applied, read off its client class.

    Looked up by name for the same reason :data:`_BASE_MEDIA_CLASSES` is: these
    classes are imported lazily so an SDK that predates one of them degrades to
    a clear error at call time rather than an ImportError at import time. The
    API-key rail needs the values because it builds the request bodies the SDK
    would otherwise have built, and a default that drifts upstream must drift
    here too rather than being frozen into a copy.
    """
    import blockrun_llm

    cls = getattr(blockrun_llm, cls_name, None)
    value = getattr(cls, attr, None) if cls is not None else None
    return value if isinstance(value, str) else None


def _get_media_client(medium: str, api_url: Optional[str], private_key: Optional[str]) -> Any:
    """Dedicated Base client for ``medium``, or the unified SolanaLLMClient
    (which get_image_client already builds + caches) when the URL is Solana."""
    if _is_solana_url(api_url):
        return get_image_client(api_url=api_url, private_key=private_key)
    import blockrun_llm

    base_cls = getattr(blockrun_llm, _BASE_MEDIA_CLASSES[medium])
    key = f"{base_cls.__name__}::{_client_key(api_url, private_key)}"
    with _lock:
        client = _media_clients.get(key)
        if client is None:
            client = base_cls(private_key=private_key, api_url=resolve_api_url(api_url))
            _media_clients[key] = client
        return client


def _is_solana_client(client: Any) -> bool:
    return _HAS_SOLANA and SolanaLLMClient is not None and isinstance(client, SolanaLLMClient)


def _solana_media_method(client: Any, method: str) -> Any:
    """Resolve a Solana media method, failing with a clear 501 instead of an
    AttributeError-500 when the installed blockrun-llm predates Solana media
    support (SolanaLLMClient.video/music/speech/sound_effect)."""
    fn = getattr(client, method, None)
    if fn is None:
        raise APIError(
            f"Solana {method} generation requires a blockrun-llm release with "
            f"SolanaLLMClient.{method} support. Upgrade: pip install -U 'blockrun-llm[solana]'",
            501,
        )
    return fn


def get_video_client(api_url: Optional[str] = None, private_key: Optional[str] = None) -> Any:
    """VideoClient (Base) or the unified SolanaLLMClient (Solana)."""
    return _get_media_client("video", api_url, private_key)


def get_music_client(api_url: Optional[str] = None, private_key: Optional[str] = None) -> Any:
    """MusicClient (Base) or the unified SolanaLLMClient (Solana)."""
    return _get_media_client("music", api_url, private_key)


def get_speech_client(api_url: Optional[str] = None, private_key: Optional[str] = None) -> Any:
    """SpeechClient (Base) or the unified SolanaLLMClient (Solana). Serves both
    TTS (speech) and sound-effects."""
    return _get_media_client("speech", api_url, private_key)


async def _run_media(
    func: Any,
    *,
    executor: concurrent.futures.ThreadPoolExecutor = _image_executor,
    ceiling: float = _SPEECH_AWAIT_CEILING_S,
) -> Any:
    """Run a sync SDK media call in a worker thread, bounded by ``ceiling``."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(executor, func), timeout=ceiling)
    except asyncio.TimeoutError:
        raise APIError(
            f"media generation exceeded the {ceiling:.0f}s server ceiling and the "
            "request was abandoned; the background job may still complete (and "
            "settle payment) — check your wallet history before retrying",
            504,
        )


# Accepted /v1/videos/generations body params, forwarded to the SDK when
# present. Single source of truth — proxy.py imports this.
VIDEO_PARAM_KEYS = (
    "image_url",
    "last_frame_url",
    "reference_image_urls",
    "real_face_asset_id",
    # Declared seed mode, cross-checked by the gateway against the seed fields
    # above (400, unbilled, on disagreement). Needs blockrun-llm >=1.7.0 on both
    # chains — see the floor in pyproject.toml.
    "input_type",
    "duration_seconds",
    "aspect_ratio",
    "resolution",
    "generate_audio",
    "seed",
    "watermark",
    "return_last_frame",
    "budget_seconds",
    "timeout",
)


async def video_generation_async(
    prompt: str,
    *,
    model: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
    **params: Any,
) -> Dict[str, Any]:
    """Generate a video. Extra kwargs (see :data:`VIDEO_PARAM_KEYS`) forward to
    the SDK. ``timeout`` is only honored on Solana (Base VideoClient has no such
    arg). Client-supplied ``budget_seconds``/``timeout`` are clamped to the
    server cap so a request body can't pin a worker thread indefinitely; a
    malformed (non-numeric) value raises ValueError → HTTP 400 at the proxy."""
    key = _route_key(api_key, private_key)
    model = _canonical_video_model(model)
    params = {k: v for k, v in params.items() if v is not None}
    for knob in ("budget_seconds", "timeout"):
        if knob in params:
            params[knob] = min(float(params[knob]), _VIDEO_BUDGET_CAP_S)
    if key:
        # ``budget_seconds``/``timeout`` govern how long WE wait, not what the
        # gateway is asked to do, so they steer the poll loop instead of riding
        # along in the body — which is what the SDK does with them too.
        budget = params.pop("budget_seconds", None) or params.pop("timeout", None)
        params.pop("timeout", None)
        body = {"model": model, "prompt": prompt, **params}
        return await _run_media(
            lambda: _apikey.submit_and_poll_video(body, api_key=key, budget_seconds=budget),
            executor=_long_media_executor,
            ceiling=_VIDEO_AWAIT_CEILING_S,
        )
    client = get_video_client(api_url=api_url, private_key=private_key)
    if _is_solana_client(client):
        video = _solana_media_method(client, "video")
        response = await _run_media(
            lambda: video(prompt, model=model, **params),
            executor=_long_media_executor,
            ceiling=_VIDEO_AWAIT_CEILING_S,
        )
    else:
        params.pop("timeout", None)  # Base VideoClient.generate has no timeout kwarg
        response = await _run_media(
            lambda: client.generate(prompt, model=model, **params),
            executor=_long_media_executor,
            ceiling=_VIDEO_AWAIT_CEILING_S,
        )
    return response.model_dump(exclude_none=True)


async def music_generation_async(
    prompt: str,
    *,
    model: Optional[str] = None,
    instrumental: bool = True,
    lyrics: Optional[str] = None,
    api_url: Optional[str] = None,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a music track. Raises ValueError (→ HTTP 400 at the proxy) when
    ``lyrics`` is combined with ``instrumental=True`` — the SDK rejects that."""
    if instrumental and lyrics and lyrics.strip():
        raise ValueError("Cannot specify lyrics when instrumental is True")
    key = _route_key(api_key, private_key)
    if key:
        body: Dict[str, Any] = {
            "model": model or _sdk_default("MusicClient", "DEFAULT_MODEL"),
            "prompt": prompt,
            "instrumental": instrumental,
        }
        if lyrics and lyrics.strip():
            body["lyrics"] = lyrics.strip()
        return await _run_media(
            lambda: _apikey.post_json(
                "/v1/audio/generations", body, api_key=key, timeout=_MUSIC_AWAIT_CEILING_S
            ),
            executor=_long_media_executor,
            ceiling=_MUSIC_AWAIT_CEILING_S,
        )
    client = get_music_client(api_url=api_url, private_key=private_key)
    # Same call shape on both chains; only the method name differs.
    media_fn = (
        _solana_media_method(client, "music") if _is_solana_client(client) else client.generate
    )
    response = await _run_media(
        lambda: media_fn(prompt, model=model, instrumental=instrumental, lyrics=lyrics),
        executor=_long_media_executor,
        ceiling=_MUSIC_AWAIT_CEILING_S,
    )
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
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize speech (TTS)."""
    key = _route_key(api_key, private_key)
    kw = {"model": model, "voice": voice, "response_format": response_format, "speed": speed}
    kw = {k: v for k, v in kw.items() if v is not None}
    if key:
        body = {
            "model": model or _sdk_default("SpeechClient", "DEFAULT_MODEL"),
            "input": input,
            **kw,
        }
        return await _run_media(
            lambda: _apikey.post_json(
                "/v1/audio/speech", body, api_key=key, timeout=_SPEECH_AWAIT_CEILING_S
            ),
            ceiling=_SPEECH_AWAIT_CEILING_S,
        )
    client = get_speech_client(api_url=api_url, private_key=private_key)
    media_fn = (
        _solana_media_method(client, "speech") if _is_solana_client(client) else client.generate
    )
    response = await _run_media(lambda: media_fn(input, **kw), ceiling=_SPEECH_AWAIT_CEILING_S)
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
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a cinematic sound effect."""
    key = _route_key(api_key, private_key)
    kw = {
        "model": model,
        "duration_seconds": duration_seconds,
        "prompt_influence": prompt_influence,
        "response_format": response_format,
    }
    kw = {k: v for k, v in kw.items() if v is not None}
    if key:
        body = {
            "model": model or _sdk_default("SpeechClient", "DEFAULT_SOUNDFX_MODEL"),
            "text": text,
            **kw,
        }
        return await _run_media(
            lambda: _apikey.post_json(
                "/v1/audio/sound-effects", body, api_key=key, timeout=_SPEECH_AWAIT_CEILING_S
            ),
            ceiling=_SPEECH_AWAIT_CEILING_S,
        )
    client = get_speech_client(api_url=api_url, private_key=private_key)
    # Both SolanaLLMClient and SpeechClient expose .sound_effect with the same
    # shape; the guard only matters on SDK versions predating Solana media.
    if _is_solana_client(client):
        sound_effect = _solana_media_method(client, "sound_effect")
    else:
        sound_effect = client.sound_effect
    response = await _run_media(lambda: sound_effect(text, **kw), ceiling=_SPEECH_AWAIT_CEILING_S)
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
    "image_edit_sync",
    "image_edit_async",
    "get_video_client",
    "get_music_client",
    "get_speech_client",
    "VIDEO_PARAM_KEYS",
    "video_generation_async",
    "music_generation_async",
    "speech_generation_async",
    "sound_effect_async",
    "APIError",
    "PaymentError",
]
