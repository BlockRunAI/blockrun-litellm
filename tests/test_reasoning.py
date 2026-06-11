"""Streaming ``reasoning_content`` passthrough.

The gateway emits reasoning-model output on the streamed delta
(``choices[0].delta.reasoning_content``) for thinking-enabled Claude (Bedrock /
direct Anthropic) and native reasoning models (DeepSeek R1, GLM thinking, ...).

LiteLLM's ``GenericStreamingChunk`` TypedDict has no reasoning field, and the
custom-provider streaming handler never promotes it onto ``delta.reasoning_content``.
The only channel a CustomLLM provider has to a live streaming consumer is
``provider_specific_fields`` (surfaced on ``delta.provider_specific_fields``).
``_to_generic_chunk`` must route the delta's reasoning there, or the reasoning
is dropped on the floor for every streamed call.

Non-stream reasoning is already covered by ``test_fingerprint`` (it survives
verbatim on ``message.reasoning_content`` through ``_build_response``).
"""

from __future__ import annotations

from blockrun_llm.types import (
    ChatChunkChoice,
    ChatChunkDelta,
    ChatCompletionChunk,
)

from blockrun_litellm.provider import _to_generic_chunk


def _chunk(delta: ChatChunkDelta, finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-reason-1",
        object="chat.completion.chunk",
        created=1_700_000_000,
        model="anthropic/claude-sonnet-4.5",
        choices=[ChatChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


def test_reasoning_delta_surfaces_in_provider_specific_fields() -> None:
    gchunk = _to_generic_chunk(
        _chunk(ChatChunkDelta(role="assistant", reasoning_content="let me think"))
    )
    psf = gchunk["provider_specific_fields"]
    assert psf is not None
    assert psf["reasoning_content"] == "let me think"


def test_content_delta_without_reasoning_has_no_reasoning_key() -> None:
    gchunk = _to_generic_chunk(_chunk(ChatChunkDelta(content="hello")))
    psf = gchunk["provider_specific_fields"]
    # A plain content delta must not invent a reasoning key.
    assert psf is None or "reasoning_content" not in psf


def test_reasoning_and_content_can_coexist_on_one_delta() -> None:
    gchunk = _to_generic_chunk(
        _chunk(ChatChunkDelta(content="answer", reasoning_content="thinking"))
    )
    assert gchunk["text"] == "answer"
    assert gchunk["provider_specific_fields"]["reasoning_content"] == "thinking"


def test_reasoning_survives_live_custom_stream_wrapper() -> None:
    """End-to-end: a reasoning delta routed through LiteLLM's real
    CustomStreamWrapper must remain readable on the live delta's
    provider_specific_fields (the only CustomLLM channel for it)."""
    import time

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.utils import custom_llm_setup

    from blockrun_litellm.provider import register

    register()
    custom_llm_setup()

    raw = [
        _chunk(ChatChunkDelta(role="assistant", reasoning_content="let me think")),
        _chunk(ChatChunkDelta(content="Hello")),
        _chunk(ChatChunkDelta(), finish_reason="stop"),
    ]
    logging_obj = Logging(
        model="blockrun/anthropic/claude-sonnet-4.5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="completion",
        litellm_call_id="test-reasoning",
        start_time=time.time(),
        function_id="test-reasoning",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=iter(_to_generic_chunk(c) for c in raw),
        model="blockrun/anthropic/claude-sonnet-4.5",
        custom_llm_provider="blockrun",
        logging_obj=logging_obj,
    )
    reasoning_seen = None
    for out in wrapper:
        psf = out.choices[0].delta.provider_specific_fields
        if psf and psf.get("reasoning_content"):
            reasoning_seen = psf["reasoning_content"]
    assert reasoning_seen == "let me think"
