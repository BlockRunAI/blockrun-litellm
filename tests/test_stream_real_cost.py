"""Streaming counterpart to ``test_real_cost.py``.

The real x402 charge must surface on **streamed** calls too, not just the
non-streaming in-process path. The SDK (blockrun-llm) attaches the per-call
charge to each stream chunk as ``chunk.cost_usd`` (race-free — it rides on the
per-call chunk object, not the shared ``client._last_call_cost``). The provider
threads that onto the *assembled* streamed response so:

  (a) LiteLLM's ``response_cost`` (spend / ``max_budget``) is the real wallet
      deduction. Older LiteLLM reads it from
      ``additional_headers['llm_provider-x-litellm-response-cost']``; newer
      LiteLLM (which takes the assembled hidden params from its own synthesized
      finish chunk) reads the usage frame's ``usage.cost``. The tests assert
      through ``Logging.response_cost_calculator``, the call the streaming
      success path makes, so they hold on either.
  (b) the JSONL audit records it with ``cost_source='blockrun_x402'``.

These tests construct chunks carrying ``cost_usd`` (simulating the SDK) and run
them through LiteLLM's real ``CustomStreamWrapper`` + ``stream_chunk_builder``,
asserting the cost survives aggregation — which, per the issue, drops cache
fields and never recomputes a provider charge on its own.
"""

from __future__ import annotations

import inspect
import time
from unittest.mock import MagicMock

import litellm
import pytest

from blockrun_llm.types import (
    ChatChunkChoice,
    ChatChunkDelta,
    ChatCompletionChunk,
    ChatUsage,
)

from blockrun_litellm import BlockRunLLM
from blockrun_litellm.logger import _build_entry

REAL_COST = 0.06354  # what the wallet was actually debited
COST_HEADER = "llm_provider-x-litellm-response-cost"

MODEL = "openai/gpt-5.5"
MESSAGES = [{"role": "user", "content": "hi"}]


def _chunk(choices, usage=None, *, cost_usd=None) -> ChatCompletionChunk:
    c = ChatCompletionChunk(
        id="chatcmpl-stream-cost-1",
        object="chat.completion.chunk",
        created=1_700_000_000,
        model=MODEL,
        choices=choices,
        usage=usage,
    )
    if cost_usd is not None:
        # The SDK attaches the per-call charge to every chunk (extra="allow").
        c.cost_usd = cost_usd
    return c


def _raw_stream(*, cost_usd):
    """content delta -> finish chunk -> include_usage final frame, each
    carrying the per-call cost_usd the SDK now attaches."""
    return [
        _chunk(
            [
                ChatChunkChoice(
                    index=0,
                    delta=ChatChunkDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
            cost_usd=cost_usd,
        ),
        _chunk(
            [ChatChunkChoice(index=0, delta=ChatChunkDelta(), finish_reason="stop")],
            cost_usd=cost_usd,
        ),
        _chunk(
            [],
            usage=ChatUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            cost_usd=cost_usd,
        ),
    ]


def _stub_stream(monkeypatch, chunks) -> MagicMock:
    mock = MagicMock()
    mock.chat_completion_stream.return_value = iter(chunks)
    monkeypatch.setattr("blockrun_litellm._adapter.get_sync_client", lambda **_: mock)
    return mock


def _real_cost(hidden_params):
    """The audit-facing charge on an assembled response: top-level
    ``blockrun_cost_usd`` (older LiteLLM) or the copy inside the propagated
    ``provider_specific_fields`` (newer LiteLLM)."""
    if "blockrun_cost_usd" in hidden_params:
        return hidden_params["blockrun_cost_usd"]
    return (hidden_params.get("provider_specific_fields") or {}).get("blockrun_cost_usd")


def _logged_cost(logging_obj, built):
    """``response_cost`` as the streaming success path computes it. The method
    is public (``response_cost_calculator``) on newer LiteLLM and private
    (``_response_cost_calculator``) on older releases."""
    calc = getattr(logging_obj, "response_cost_calculator", None) or getattr(
        logging_obj, "_response_cost_calculator"
    )
    return calc(result=built)


def _assemble(monkeypatch, chunks, *, with_logging=False):
    """Drive the provider's streaming() through LiteLLM's real
    CustomStreamWrapper and return the aggregated ModelResponse (and the
    ``Logging`` object, when ``with_logging``)."""
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.utils import custom_llm_setup

    _stub_stream(monkeypatch, chunks)
    BlockRunLLM()  # ensure provider importable
    from blockrun_litellm import register

    register()
    custom_llm_setup()

    gen = BlockRunLLM().streaming(
        model=MODEL, messages=MESSAGES, optional_params={"max_tokens": 32}
    )
    logging_obj = Logging(
        model=f"blockrun/{MODEL}",
        messages=MESSAGES,
        stream=True,
        call_type="completion",
        litellm_call_id="test-stream-cost",
        start_time=time.time(),
        function_id="test-stream-cost",
    )
    # What ``litellm.completion`` does before calling the provider; without it
    # the Logging object has no litellm_params and cannot price the result.
    logging_obj.update_environment_variables(
        model=f"blockrun/{MODEL}",
        user=None,
        optional_params={},
        litellm_params={"custom_llm_provider": "blockrun"},
        custom_llm_provider="blockrun",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=gen,
        model=f"blockrun/{MODEL}",
        logging_obj=logging_obj,
        custom_llm_provider="blockrun",
    )
    list(wrapper)
    builder_kwargs = {}
    if "logging_obj" in inspect.signature(litellm.stream_chunk_builder).parameters:
        # Newer LiteLLM resolves the stream's response_cost inside the builder
        # and needs the Logging object to do it; older releases do not take it.
        builder_kwargs["logging_obj"] = logging_obj
    built = litellm.stream_chunk_builder(
        chunks=wrapper.chunks, messages=MESSAGES, **builder_kwargs
    )
    return (built, logging_obj) if with_logging else built


def test_streaming_real_cost_survives_aggregation(monkeypatch) -> None:
    built = _assemble(monkeypatch, _raw_stream(cost_usd=REAL_COST))
    assert built is not None
    # Token usage still flows from the gateway's real counts.
    assert built.usage.prompt_tokens == 100
    assert built.usage.completion_tokens == 20
    # (b) audit-facing field
    assert _real_cost(built._hidden_params) == pytest.approx(REAL_COST)


def test_streaming_litellm_response_cost_uses_real_charge(monkeypatch) -> None:
    # The streaming success path sets ``response_cost`` from
    # ``Logging.response_cost_calculator(result=<assembled response>)``. It
    # must return the real wallet charge, not a token x list-price estimate
    # (and must not depend on the model being in LiteLLM's price map).
    built, logging_obj = _assemble(
        monkeypatch, _raw_stream(cost_usd=REAL_COST), with_logging=True
    )
    assert _logged_cost(logging_obj, built) == pytest.approx(REAL_COST)


def test_streaming_logger_records_real_cost_and_source(monkeypatch) -> None:
    built = _assemble(monkeypatch, _raw_stream(cost_usd=REAL_COST))
    kwargs = {"model": f"blockrun/{MODEL}", "messages": MESSAGES, "stream": True}
    entry = _build_entry(kwargs, built, time.time(), time.time())
    assert entry is not None
    assert entry["cost_usd"] == pytest.approx(REAL_COST)
    assert entry["cost_source"] == "blockrun_x402"
    assert entry["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }


def test_streaming_free_call_records_zero_not_stale(monkeypatch) -> None:
    # Paid-path SDK attaches cost_usd=0.0 for free/cached calls; surface a
    # race-free zero, not a stale prior charge.
    built, logging_obj = _assemble(monkeypatch, _raw_stream(cost_usd=0.0), with_logging=True)
    assert _real_cost(built._hidden_params) == 0.0
    assert _logged_cost(logging_obj, built) == 0.0
    entry = _build_entry(
        {"model": f"blockrun/{MODEL}", "stream": True, "messages": MESSAGES},
        built,
        time.time(),
        time.time(),
    )
    assert entry["cost_usd"] == 0.0
    assert entry["cost_source"] == "blockrun_x402"


def test_streaming_without_sdk_cost_falls_back_to_estimate(monkeypatch) -> None:
    # Older SDK: no cost_usd on chunks -> no real cost, no clobbered hidden params.
    built = _assemble(monkeypatch, _raw_stream(cost_usd=None))
    assert _real_cost(built._hidden_params) is None
    assert getattr(built.usage, "cost", None) != REAL_COST
    assert COST_HEADER not in built._hidden_params.get("additional_headers", {})
