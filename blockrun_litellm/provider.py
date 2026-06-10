"""
LiteLLM ``CustomLLM`` handler for BlockRun.

Usage
-----
::

    import litellm
    from blockrun_litellm import register

    register()  # adds the "blockrun" provider to LiteLLM

    response = litellm.completion(
        model="blockrun/openai/gpt-5.5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=128,
    )
    print(response.choices[0].message.content)

The ``register()`` call appends an entry to ``litellm.custom_provider_map``.
Calling it twice is idempotent.

Wallet
------
The underlying ``blockrun-llm`` SDK reads the private key from (in order):

1. ``private_key`` kwarg forwarded via ``optional_params``
2. ``BLOCKRUN_WALLET_KEY`` env var
3. ``BASE_CHAIN_WALLET_KEY`` env var
4. ``~/.blockrun/.session`` (created by ``setup_agent_wallet()``)

The key never leaves the host — only EIP-712 signatures travel over the wire.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk

from blockrun_llm.types import APIError as BlockRunAPIError
from blockrun_llm.types import ChatCompletionChunk

from blockrun_litellm import _adapter


# Provider name surfaced to LiteLLM exception classes so the router knows
# this is BlockRun-routed when it logs / retries / falls back.
_LITELLM_PROVIDER = "blockrun"


def _translate_to_litellm(exc: Exception, model: str) -> Optional[Exception]:
    """Map a transient BlockRun / network error to LiteLLM's retriable
    exception hierarchy so the router's own fallback machinery kicks in.

    Returns the LiteLLM-compatible exception to raise, or ``None`` if the
    exception is not transient (caller should re-raise as-is).

    Mappings:
      * ``httpx.TimeoutException``        → ``litellm.Timeout``
      * ``httpx.NetworkError``            → ``litellm.APIConnectionError``
      * ``APIError`` 500                  → ``litellm.InternalServerError``
      * ``APIError`` 502 / 504            → ``litellm.APIConnectionError``
      * ``APIError`` 503                  → ``litellm.ServiceUnavailableError``
      * ``APIError`` 429                  → ``litellm.RateLimitError``
    """
    if isinstance(exc, httpx.TimeoutException):
        return litellm.Timeout(
            message=f"BlockRun upstream timed out: {exc}",
            model=model,
            llm_provider=_LITELLM_PROVIDER,
        )
    if isinstance(exc, httpx.NetworkError):
        return litellm.APIConnectionError(
            message=f"BlockRun upstream network error: {exc}",
            model=model,
            llm_provider=_LITELLM_PROVIDER,
        )
    if isinstance(exc, BlockRunAPIError):
        status = getattr(exc, "status_code", 0)
        if status == 429:
            return litellm.RateLimitError(
                message=str(exc), model=model, llm_provider=_LITELLM_PROVIDER
            )
        if status == 500:
            return litellm.InternalServerError(
                message=str(exc), model=model, llm_provider=_LITELLM_PROVIDER
            )
        if status == 503:
            return litellm.ServiceUnavailableError(
                message=str(exc), model=model, llm_provider=_LITELLM_PROVIDER
            )
        if status in (502, 504):
            return litellm.APIConnectionError(
                message=str(exc), model=model, llm_provider=_LITELLM_PROVIDER
            )
    return None


# LiteLLM passes the provider-stripped model name *and* an "optional_params"
# dict containing the OpenAI-style params (temperature, max_tokens, ...).
# It also forwards ``api_base`` / ``api_key`` from the call site, which we
# repurpose: ``api_base`` → BlockRun ``api_url``, ``api_key`` → wallet key.
_OPTIONAL_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "stream",
)


def _collect_openai_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    optional = kwargs.get("optional_params") or {}
    for k in _OPTIONAL_KEYS:
        if k in optional and optional[k] is not None:
            out[k] = optional[k]
        elif k in kwargs and kwargs[k] is not None:
            out[k] = kwargs[k]
    # BlockRun-specific extras may be passed via ``litellm_params`` /
    # ``optional_params``. Surface them if present.
    for k in ("search", "search_parameters", "fallback_models"):
        if k in optional and optional[k] is not None:
            out[k] = optional[k]
    return out


def _build_response(model: str, payload: Dict[str, Any]) -> litellm.ModelResponse:
    """Wrap a BlockRun-dumped dict into a ``litellm.ModelResponse``.

    Native-fingerprint passthrough: the gateway returns the upstream
    provider's response verbatim, so the dumped dict carries the real
    relay-detection signals — ``system_fingerprint`` (GPT ``fp_*``),
    ``service_tier``, ``usage.prompt_tokens_details`` /
    ``usage.cache_read_input_tokens`` / ``cache_creation_input_tokens``,
    and per-message ``reasoning_content``. ``litellm.ModelResponse``
    preserves all of these as first-class or extra fields, so a relay
    detector (e.g. cctest.ai) sees a genuine direct upstream call. The
    regression suite in ``tests/test_fingerprint.py`` locks this in so a
    future LiteLLM / SDK bump can't silently strip them.
    """
    # The payload from blockrun-llm already includes the model field. We pop
    # it to avoid the duplicate-kwarg TypeError, then re-inject the
    # caller-supplied id so callers see the model they actually asked for
    # (BlockRun may rewrite e.g. "openai/gpt-5.5" → bare "gpt-5.5").
    payload = dict(payload)
    payload.pop("model", None)
    return litellm.ModelResponse(**payload, model=model)


def _native_extras(chunk: ChatCompletionChunk) -> Dict[str, Any]:
    """Collect the upstream-native fingerprint fields carried on a stream
    chunk (e.g. ``system_fingerprint``, ``service_tier``) so they survive
    the lossy :class:`GenericStreamingChunk` contract.

    ``ChatCompletionChunk`` is declared ``extra = "allow"`` in blockrun-llm,
    so any top-level field the gateway forwards that isn't part of the
    OpenAI chunk schema lands in ``model_extra``. We surface those plus the
    usage-level cache/details extras through ``provider_specific_fields`` so
    in-process LiteLLM streaming callers can still read the genuine signals.
    """
    extras: Dict[str, Any] = dict(chunk.model_extra or {})
    if chunk.usage is not None:
        usage_extra = chunk.usage.model_extra or {}
        if usage_extra:
            extras.setdefault("usage_details", {}).update(usage_extra)
    return extras


def _to_generic_chunk(chunk: ChatCompletionChunk) -> GenericStreamingChunk:
    """Map a BlockRun :class:`ChatCompletionChunk` → LiteLLM
    :class:`GenericStreamingChunk` (a ``TypedDict``).

    BlockRun's chunk schema is the OpenAI ``chat.completion.chunk`` schema.
    LiteLLM's ``GenericStreamingChunk`` is a simpler, provider-agnostic
    structure that the streaming handler stitches into a final response.
    """
    # Native fingerprint fields ride on every chunk (OpenAI sends
    # ``system_fingerprint`` per chunk), so collect them regardless of whether
    # this chunk carries a choice, and surface via ``provider_specific_fields``.
    extras = _native_extras(chunk)
    provider_specific_fields = extras or None

    if not chunk.choices:
        # A choice-less chunk is the OpenAI `include_usage` final frame
        # (choices:[] + usage). Forward its real token counts so LiteLLM bills
        # off them instead of re-estimating the prompt with its own tokenizer
        # (tiktoken drifts ~37% vs the gateway's real upstream count). Older
        # gateways that never send this frame still hit the usage=None path.
        usage = None
        if chunk.usage is not None:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
        return GenericStreamingChunk(
            text="",
            is_finished=False,
            finish_reason="",
            usage=usage,
            index=0,
            provider_specific_fields=provider_specific_fields,
        )

    choice = chunk.choices[0]
    text = choice.delta.content or ""
    finish_reason = choice.finish_reason or ""

    # BlockRun's per-chunk usage is rarely populated; LiteLLM tolerates None.
    usage = None
    if chunk.usage is not None:
        usage = {
            "prompt_tokens": chunk.usage.prompt_tokens,
            "completion_tokens": chunk.usage.completion_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }

    return GenericStreamingChunk(
        text=text,
        is_finished=bool(finish_reason),
        finish_reason=finish_reason,
        usage=usage,
        index=choice.index,
        provider_specific_fields=provider_specific_fields,
    )


class BlockRunLLM(CustomLLM):
    """LiteLLM custom provider that routes through BlockRun's x402 gateway."""

    # NOTE: signature must match litellm.CustomLLM exactly. LiteLLM passes a
    # *lot* of kwargs; we only consume what we recognize.
    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> litellm.ModelResponse:
        openai_kwargs = _collect_openai_kwargs(kwargs)
        try:
            payload = _adapter.chat_completion_sync(
                model=model,
                messages=messages,
                api_url=api_base,
                private_key=api_key,
                **openai_kwargs,
            )
        except Exception as exc:
            translated = _translate_to_litellm(exc, model)
            if translated is not None:
                raise translated from exc
            raise
        return _build_response(model, payload)

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> litellm.ModelResponse:
        openai_kwargs = _collect_openai_kwargs(kwargs)
        try:
            payload = await _adapter.chat_completion_async(
                model=model,
                messages=messages,
                api_url=api_base,
                private_key=api_key,
                **openai_kwargs,
            )
        except Exception as exc:
            translated = _translate_to_litellm(exc, model)
            if translated is not None:
                raise translated from exc
            raise
        return _build_response(model, payload)

    # ----- Streaming -------------------------------------------------------

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        """Sync streaming. Yields :class:`GenericStreamingChunk` per delta.

        Errors from the SDK during stream setup or mid-flight are
        translated to LiteLLM's retriable exception types via
        :func:`_translate_to_litellm` so the router's own fallback
        machinery can pick the next provider on transient upstream issues.
        """
        openai_kwargs = _collect_openai_kwargs(kwargs)
        # ``stream`` is implicit at this entrypoint; drop it so the SDK doesn't
        # see a duplicate kwarg.
        openai_kwargs.pop("stream", None)
        try:
            for chunk in _adapter.chat_completion_stream_sync(
                model=model,
                messages=messages,
                api_url=api_base,
                private_key=api_key,
                **openai_kwargs,
            ):
                yield _to_generic_chunk(chunk)
        except Exception as exc:
            translated = _translate_to_litellm(exc, model)
            if translated is not None:
                raise translated from exc
            raise

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming. Same semantics as :meth:`streaming`."""
        openai_kwargs = _collect_openai_kwargs(kwargs)
        openai_kwargs.pop("stream", None)
        try:
            async for chunk in _adapter.chat_completion_stream_async(
                model=model,
                messages=messages,
                api_url=api_base,
                private_key=api_key,
                **openai_kwargs,
            ):
                yield _to_generic_chunk(chunk)
        except Exception as exc:
            translated = _translate_to_litellm(exc, model)
            if translated is not None:
                raise translated from exc
            raise


_PROVIDER_NAME = "blockrun"
_handler: Optional[BlockRunLLM] = None


def register() -> BlockRunLLM:
    """
    Register the ``blockrun`` provider with LiteLLM.

    Idempotent — calling twice does not duplicate the entry.

    Returns the singleton handler instance so callers can swap in a custom
    subclass via ``litellm.custom_provider_map`` if they need to.
    """
    global _handler
    if _handler is None:
        _handler = BlockRunLLM()

    existing = getattr(litellm, "custom_provider_map", None) or []
    for entry in existing:
        if isinstance(entry, dict) and entry.get("provider") == _PROVIDER_NAME:
            return _handler

    existing.append({"provider": _PROVIDER_NAME, "custom_handler": _handler})
    litellm.custom_provider_map = existing
    return _handler
