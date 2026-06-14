"""Streaming tool calls must reach the Anthropic /v1/messages bridge with args.

BlockRun's SDK delivers each tool call complete (id + name + full arguments) in
one chunk. LiteLLM's custom-provider → Anthropic adapter is a state machine that
expects OpenAI-style incremental streaming: a frame with the function *name*
opens the ``tool_use`` content block, a later frame with only *arguments*
streams them as ``input_json_delta``. Handed name+args together it drops the
args (tool_use block with empty input) — so ``_iter_stream_chunks`` splits each
tool call into a name frame followed by an arguments frame.

Without this, agentic clients (Claude Code via /v1/messages) get tool calls with
no input and can't act — plain chat streams fine, every tool step is empty.
"""

import asyncio
from types import SimpleNamespace as NS

import pytest

from blockrun_litellm.provider import _iter_stream_chunks


def _litellm_anthropic_accepts_blockrun() -> bool:
    """Does the installed litellm's experimental /v1/messages handler accept our
    custom provider?

    ``register()`` adds ``blockrun`` to ``litellm.custom_provider_map``, but the
    anthropic-messages handler (litellm>=~1.61) instead validates the provider
    against the ``LlmProviders`` enum — which ``blockrun`` is not in — and raises
    ``ValueError`` before our adapter ever runs. On those versions the end-to-end
    translation path below cannot execute, so we skip it rather than hard-fail.
    The unit tests above (``_iter_stream_chunks``) cover the split/index logic and
    are unaffected.
    """
    import litellm

    try:
        litellm.LlmProviders("blockrun")
        return True
    except ValueError:
        return False


def _chunk(delta, *, finish_reason=None, usage=None):
    choice = NS(index=0, delta=delta, finish_reason=finish_reason)
    return NS(choices=[choice], usage=usage, model_extra=None)


def _delta(*, content=None, tool_calls=None, reasoning_content=None):
    return NS(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


def _tc(*, id=None, name=None, arguments=None):
    # Mirror production blockrun_llm.types.ToolCall: id/type/function only — NO
    # `index` field. The stream-scoped counter in _iter_stream_chunks must assign
    # the content-block index; it must not read one off the tool call.
    return NS(id=id, type="function", function=NS(name=name, arguments=arguments))


# ---------------------------------------------------------------------------
# Unit: a complete tool call is split into a name frame + an arguments frame.
# ---------------------------------------------------------------------------

def test_tool_call_split_into_name_then_args():
    out = list(_iter_stream_chunks(
        _chunk(_delta(tool_calls=[_tc(id="call_1", name="list_files", arguments='{"path":"."}')]))
    ))
    assert len(out) == 2

    # Frame 1 — name (opens the tool_use content block); args empty.
    name_frame = out[0]["tool_use"]
    assert name_frame["id"] == "call_1"
    assert name_frame["function"]["name"] == "list_files"
    assert name_frame["function"]["arguments"] == ""

    # Frame 2 — arguments only (streamed as input_json_delta); no id/name.
    args_frame = out[1]["tool_use"]
    assert args_frame["id"] is None
    assert args_frame["function"]["name"] is None
    assert args_frame["function"]["arguments"] == '{"path":"."}'


def test_plain_text_passes_through_without_tool_use():
    out = list(_iter_stream_chunks(_chunk(_delta(content="hello"))))
    assert len(out) == 1
    assert "tool_use" not in out[0]
    assert out[0]["text"] == "hello"


def test_finish_chunk_passes_through():
    out = list(_iter_stream_chunks(_chunk(_delta(), finish_reason="stop")))
    assert len(out) == 1
    assert out[0]["finish_reason"] == "stop"


def test_tool_call_chunk_carrying_finish_also_forwards_it():
    out = list(_iter_stream_chunks(
        _chunk(_delta(tool_calls=[_tc(id="c", name="f", arguments="{}")]), finish_reason="tool_calls")
    ))
    assert out[0]["tool_use"]["function"]["name"] == "f"            # name frame
    assert out[1]["tool_use"]["function"]["arguments"] == "{}"      # args frame
    assert any(g.get("finish_reason") == "tool_calls" for g in out)  # finish forwarded


def test_parallel_tool_calls_each_get_their_own_block():
    # Two parallel calls in ONE chunk — the counter must give them distinct
    # block indices even though neither ToolCall carries an `index`.
    state = {}
    out = list(_iter_stream_chunks(_chunk(_delta(tool_calls=[
        _tc(id="a", name="f0", arguments="{}"),
        _tc(id="b", name="f1", arguments="{}"),
    ])), state))
    indices = [o["tool_use"]["index"] for o in out]
    assert indices == [0, 0, 1, 1]  # name+args per call, distinct block indices


def test_parallel_tool_calls_across_chunks_get_distinct_blocks():
    # BlockRun may deliver each parallel call in its OWN chunk. The stream-scoped
    # state dict must keep the counter monotonic across chunks, or both collapse
    # onto block 0 and the second call is lost.
    state = {}
    out0 = list(_iter_stream_chunks(_chunk(_delta(tool_calls=[_tc(id="a", name="f0", arguments="{}")])), state))
    out1 = list(_iter_stream_chunks(_chunk(_delta(tool_calls=[_tc(id="b", name="f1", arguments="{}")])), state))
    assert [o["tool_use"]["index"] for o in out0] == [0, 0]
    assert [o["tool_use"]["index"] for o in out1] == [1, 1]


def test_tool_index_does_not_leak_across_streams():
    # A fresh state dict per stream means each stream starts its tool blocks at 0.
    first = list(_iter_stream_chunks(_chunk(_delta(tool_calls=[_tc(id="a", name="f", arguments="{}")])), {}))
    second = list(_iter_stream_chunks(_chunk(_delta(tool_calls=[_tc(id="b", name="g", arguments="{}")])), {}))
    assert first[0]["tool_use"]["index"] == 0
    assert second[0]["tool_use"]["index"] == 0


# ---------------------------------------------------------------------------
# End-to-end: a streamed tool call survives the real LiteLLM /v1/messages
# (Anthropic) translation with its arguments intact.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _litellm_anthropic_accepts_blockrun(),
    reason="installed litellm validates the custom provider against the "
    "LlmProviders enum in its /v1/messages handler; 'blockrun' isn't in that "
    "enum, so this end-to-end translation path can't run on this version.",
)
def test_tool_call_reaches_anthropic_messages_with_arguments():
    from litellm.anthropic_interface.messages import acreate
    import blockrun_litellm._adapter as adapter
    from blockrun_litellm.provider import register
    from blockrun_llm.types import (
        ChatCompletionChunk, ChatChunkChoice, ChatChunkDelta, ChatUsage,
    )

    register()

    def _sdk_chunk(choices, usage=None):
        return ChatCompletionChunk(
            id="c1", object="chat.completion.chunk", created=1_700_000_000,
            model="openai/gpt-5.5", choices=choices, usage=usage,
        )

    async def fake_stream(*a, **k):
        yield _sdk_chunk([ChatChunkChoice(index=0, delta=ChatChunkDelta(role="assistant"), finish_reason=None)])
        # Build the tool call from a dict so pydantic coerces it to whichever
        # type the installed SDK declares for ChatChunkDelta.tool_calls — the
        # strict ToolCall (older SDK) or the lenient ChatChunkToolCall (the SDK's
        # streamed-tool-call fix). Keeps this test green across that SDK change.
        yield _sdk_chunk([ChatChunkChoice(index=0, delta=ChatChunkDelta(
            tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                         "function": {"name": "list_files", "arguments": '{"path":"/tmp"}'}}]
        ), finish_reason=None)])
        yield _sdk_chunk([ChatChunkChoice(index=0, delta=ChatChunkDelta(), finish_reason="tool_calls")])
        yield _sdk_chunk([], usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    original = adapter.chat_completion_stream_async
    adapter.chat_completion_stream_async = fake_stream
    try:
        async def _collect():
            resp = await acreate(
                model="blockrun/openai/gpt-5.5",
                messages=[{"role": "user", "content": "list the files"}],
                tools=[{"name": "list_files", "description": "x",
                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
                max_tokens=100, stream=True,
            )
            parts = []
            async for ev in resp:
                parts.append(ev if isinstance(ev, str) else (ev.decode() if isinstance(ev, bytes) else str(ev)))
            return "".join(parts)

        full = asyncio.run(_collect())
    finally:
        adapter.chat_completion_stream_async = original

    assert '"type": "tool_use"' in full          # tool_use content block opened
    assert "list_files" in full                  # with the function name
    assert "input_json_delta" in full            # arguments streamed
    assert "/tmp" in full                        # ...with the real argument value
    assert '"stop_reason": "tool_use"' in full   # correct stop reason
