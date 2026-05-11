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

from typing import Any, Dict, List, Optional

import litellm
from litellm import CustomLLM

from blockrun_litellm import _adapter


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
    """Wrap a BlockRun-dumped dict into a ``litellm.ModelResponse``."""
    # The payload from blockrun-llm already includes the model field. We pop
    # it to avoid the duplicate-kwarg TypeError, then re-inject the
    # caller-supplied id so callers see the model they actually asked for
    # (BlockRun may rewrite e.g. "openai/gpt-5.5" → bare "gpt-5.5").
    payload = dict(payload)
    payload.pop("model", None)
    return litellm.ModelResponse(**payload, model=model)


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
        payload = _adapter.chat_completion_sync(
            model=model,
            messages=messages,
            api_url=api_base,
            private_key=api_key,
            **openai_kwargs,
        )
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
        payload = await _adapter.chat_completion_async(
            model=model,
            messages=messages,
            api_url=api_base,
            private_key=api_key,
            **openai_kwargs,
        )
        return _build_response(model, payload)


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
