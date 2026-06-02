"""
Local OpenAI-compatible proxy for BlockRun.

Run as a sidecar; point any OpenAI client (LiteLLM, langchain, raw SDK,
curl) at ``http://localhost:4001/v1`` and it Just Works. Your x402 wallet
key lives in this process — clients on the same host never see it.

Endpoints
---------
- ``POST /v1/chat/completions``    — OpenAI Chat Completions
- ``POST /v1/images/generations``  — OpenAI Image Generations (DALL-E compatible)
- ``GET  /v1/models``              — passthrough to BlockRun's chat catalog
- ``GET  /healthz``                — liveness probe (no upstream call)

Auth
----
The proxy itself is **unauthenticated by default** — bind to ``127.0.0.1``
in production. Optional shared-secret guard via ``BLOCKRUN_PROXY_TOKEN``;
clients must then send ``Authorization: Bearer <token>``.

CLI
---
::

    $ pip install 'blockrun-litellm[proxy]'
    $ export BLOCKRUN_WALLET_KEY=0x...
    $ blockrun-litellm-proxy --port 4001
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "blockrun-litellm[proxy] extras not installed. "
        "Run: pip install 'blockrun-litellm[proxy]'"
    ) from exc

import json as _json

from blockrun_llm.types import APIError, PaymentError

from blockrun_litellm import _adapter

# Optional — present when blockrun-llm[solana] is installed alongside solana-py.
# SolanaRpcException wraps httpx / network errors from the Solana JSON-RPC layer
# (e.g. getAccountInfo failure during x402 payment signing). We handle it
# explicitly so it doesn't produce a noisy full-traceback in the log.
try:
    from solana.exceptions import SolanaRpcException as _SolanaRpcException  # type: ignore[import-untyped]
except ImportError:
    _SolanaRpcException = None  # type: ignore[assignment,misc]

log = logging.getLogger("blockrun_litellm.proxy")

# Limit concurrent in-flight requests to avoid saturating upstream rate limits.
# Anthropic (claude-opus-*) has strict TPM/RPM caps — raising this too high
# without also raising upstream quota will cause 429s.
# Override with env var BLOCKRUN_MAX_CONCURRENT (int, default 100).
# The httpx pool is configured for 200 connections; each paid request uses 2
# connections (402 probe + authenticated call), so 100 is the practical ceiling
# before the pool itself becomes the bottleneck.
_MAX_CONCURRENT: int = int(os.environ.get("BLOCKRUN_MAX_CONCURRENT", "100"))
_concurrency_sem: asyncio.Semaphore  # initialised in lifespan / lazily on first use


def _get_semaphore() -> asyncio.Semaphore:
    global _concurrency_sem
    try:
        return _concurrency_sem
    except NameError:
        _concurrency_sem = asyncio.Semaphore(_MAX_CONCURRENT)
        return _concurrency_sem


app = FastAPI(
    title="blockrun-litellm proxy",
    description="OpenAI-compatible front-end for BlockRun's x402 gateway.",
    version="0.2.0",
    docs_url="/docs",
)


# ---------------------------------------------------------------------------
# Optional shared-secret auth
# ---------------------------------------------------------------------------

def _expected_token() -> Optional[str]:
    return os.environ.get("BLOCKRUN_PROXY_TOKEN") or None


def _require_token(authorization: Optional[str] = Header(default=None)) -> None:
    expected = _expected_token()
    if expected is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(401, "Invalid Bearer token")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(_require_token)])
async def list_models() -> Dict[str, Any]:
    # Reuse the cached async client; ``list_models`` does not require payment.
    client = _adapter.get_async_client()
    models = await client.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.get("id"),
                "object": "model",
                "owned_by": m.get("provider", "blockrun"),
            }
            for m in models
        ],
    }


def _is_solana_rpc_exc(exc: Exception) -> bool:
    """Return True when ``exc`` is a ``SolanaRpcException`` (optional dep)."""
    return _SolanaRpcException is not None and isinstance(exc, _SolanaRpcException)


def _solana_rpc_msg(exc: Exception) -> str:
    return str(getattr(exc, "error_msg", exc))


def _payment_error_payload(exc: PaymentError) -> Dict[str, Any]:
    """Render a :class:`PaymentError` into the JSON body for an HTTP 402.

    When the SDK preserves the gateway's original failure reason
    (``status_code`` + ``response.details``), surface it verbatim so
    customers see the real facilitator error (e.g.
    ``transaction_simulation_failed``) instead of just our SDK's
    generic message. Falls back to ``{"error": str(exc)}`` for older
    PaymentError instances that don't carry a response dict.
    """
    payload: Dict[str, Any] = {"error": str(exc)}
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        details = resp.get("details")
        if isinstance(details, str) and details:
            payload["details"] = details
    return payload


def _payment_error_sse_message(exc: PaymentError) -> str:
    """Stream-mode equivalent of :func:`_payment_error_payload` — folds the
    gateway ``details`` into the OpenAI-style ``message`` field so a streaming
    client still sees the real reason in the single error event."""
    msg = str(exc)
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        details = resp.get("details")
        if isinstance(details, str) and details and details not in msg:
            msg = f"{msg} (details: {details})"
    return msg


def _openai_error_event(message: str, code: int = 500) -> str:
    """Render an error as an SSE ``data:`` line in OpenAI's error schema.

    The openai Python SDK (v1.x, Rust parser) recognises ``{"error": {...}}``
    streaming events only when the nested object uses OpenAI's exact field
    names: ``message``, ``type``, ``param``, ``code``.  Non-conforming shapes
    (e.g. a ``status`` key instead of ``code``) cause the Rust untagged-enum
    parser to fail with "data did not match any variant", which LiteLLM then
    wraps as a confusing ``MidStreamFallbackError``.
    """
    payload = {
        "error": {
            "message": message,
            "type": "upstream_error",
            "param": None,
            "code": str(code),
        }
    }
    return f"data: {_json.dumps(payload)}\n\n"


async def _sse_event_stream(
    model: str,
    messages: List[Dict[str, Any]],
    openai_kwargs: Dict[str, Any],
):
    """Async generator that renders BlockRun chunks as OpenAI SSE events.

    Each chunk is emitted as ``data: <json>\\n\\n`` (OpenAI convention),
    followed by a terminating ``data: [DONE]\\n\\n`` once the upstream
    iterator drains. Errors during streaming are surfaced as a final
    ``data: {"error": ...}`` event before the [DONE], since headers
    have already been flushed.

    Concurrency note: the semaphore is held only until the first chunk
    arrives from upstream — covering the x402 probe + sign + HTTP handshake.
    The remaining chunks stream freely without occupying a semaphore slot,
    so a slow client reading at 1 token/s does not block 100 other callers.
    """
    stream_gen = _adapter.chat_completion_stream_async(
        model=model, messages=messages, **openai_kwargs
    )

    # Phase 1: acquire semaphore, establish upstream connection, get first chunk.
    first_chunk_data: Optional[str] = None
    error_event: Optional[str] = None

    async with _get_semaphore():
        try:
            first_obj = await stream_gen.__anext__()
            first_chunk_data = f"data: {_json.dumps(first_obj.model_dump(exclude_none=True))}\n\n"
        except StopAsyncIteration:
            pass  # empty stream — unusual but not fatal
        except PaymentError as exc:
            error_event = _openai_error_event(_payment_error_sse_message(exc), code=402)
        except APIError as exc:
            error_event = _openai_error_event(str(exc), code=getattr(exc, "status_code", 500) or 500)
        except Exception as exc:
            if _is_solana_rpc_exc(exc):
                log.warning("solana rpc error during payment signing: %s", _solana_rpc_msg(exc))
                error_event = _openai_error_event(_solana_rpc_msg(exc), code=503)
            else:
                log.exception("stream error (first chunk)")
                error_event = _openai_error_event(str(exc), code=500)

    # Semaphore released — emit first chunk or error, then stream the rest.
    if error_event:
        yield error_event
        yield "data: [DONE]\n\n"
        return

    if first_chunk_data:
        yield first_chunk_data

    # Phase 2: drain remaining chunks without holding the semaphore.
    try:
        async for chunk in stream_gen:
            yield f"data: {_json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
    except PaymentError as exc:
        yield _openai_error_event(_payment_error_sse_message(exc), code=402)
    except APIError as exc:
        yield _openai_error_event(str(exc), code=getattr(exc, "status_code", 500) or 500)
    except Exception as exc:
        if _is_solana_rpc_exc(exc):
            log.warning("solana rpc error during stream: %s", _solana_rpc_msg(exc))
            yield _openai_error_event(_solana_rpc_msg(exc), code=503)
        else:
            log.exception("stream error")
            yield _openai_error_event(str(exc), code=500)

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(_require_token)])
async def chat_completions(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    model = body.get("model")
    messages = body.get("messages")
    if not model or not isinstance(messages, list):
        raise HTTPException(400, "`model` and `messages` are required")

    stream = bool(body.get("stream"))

    # Drop control fields before forwarding the rest as OpenAI kwargs.
    openai_kwargs = {k: v for k, v in body.items() if k not in ("model", "messages", "stream")}

    if stream:
        # SSE response. Errors before the first chunk (e.g. payment failure
        # during the 402 dance) get embedded in the stream rather than as
        # HTTP errors — easier for OpenAI-compatible clients to consume.
        return StreamingResponse(
            _sse_event_stream(model, messages, openai_kwargs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable proxy buffering
                "Connection": "keep-alive",
            },
        )

    async with _get_semaphore():
        try:
            payload = await _adapter.chat_completion_async(
                model=model,
                messages=messages,
                **openai_kwargs,
            )
        except PaymentError as exc:
            return JSONResponse(status_code=402, content=_payment_error_payload(exc))
        except APIError as exc:
            status = exc.status_code if 400 <= getattr(exc, "status_code", 0) < 600 else 502
            return JSONResponse(status_code=status, content={"error": str(exc)})
        except Exception as exc:
            if _is_solana_rpc_exc(exc):
                log.warning("solana rpc error: %s", _solana_rpc_msg(exc))
                raise HTTPException(503, _solana_rpc_msg(exc))
            raise

    return payload


@app.post("/v1/images/generations", dependencies=[Depends(_require_token)])
async def image_generations(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(400, "`prompt` is required")

    model: Optional[str] = body.get("model")
    size: Optional[str] = body.get("size")
    n: int = int(body.get("n", 1))

    async with _get_semaphore():
        try:
            result = await _adapter.image_generation_async(
                prompt=prompt,
                model=model,
                size=size,
                n=n,
            )
        except PaymentError as exc:
            return JSONResponse(status_code=402, content=_payment_error_payload(exc))
        except APIError as exc:
            status = exc.status_code if 400 <= getattr(exc, "status_code", 0) < 600 else 502
            return JSONResponse(status_code=status, content={"error": str(exc)})

    return result


# ---------------------------------------------------------------------------
# OpenAI Responses API bridge (/v1/responses)
# ---------------------------------------------------------------------------
# BlockRun's gateway only speaks Chat Completions. This endpoint accepts an
# OpenAI Responses API request (``input`` instead of ``messages``), translates
# it to a chat completion, and translates the result back to the Responses API
# shape (non-stream JSON, or the typed ``response.*`` SSE event sequence when
# ``stream=True``). Text in / text out is fully bridged; Responses-only inputs
# (tools-as-state, reasoning items, ``previous_response_id``, ``store``) are not
# round-tripped — use /v1/chat/completions for advanced tool/state flows.

_RESPONSES_INPUT_ROLES = {"system", "user", "assistant", "tool"}


def _responses_to_chat(body: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any], bool]:
    """Translate a Responses API request → (model, messages, openai_kwargs, stream)."""
    model = body.get("model")
    messages: List[Dict[str, Any]] = []

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in _RESPONSES_INPUT_ROLES:
                role = "user"
            content = item.get("content")
            if isinstance(content, list):
                # content parts: [{type: input_text|output_text|text, text: "..."}]
                content = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                )
            messages.append({"role": role, "content": content if isinstance(content, str) else ""})

    openai_kwargs: Dict[str, Any] = {}
    if body.get("max_output_tokens") is not None:
        openai_kwargs["max_tokens"] = body["max_output_tokens"]
    for k in ("temperature", "top_p", "tools", "tool_choice"):
        if body.get(k) is not None:
            openai_kwargs[k] = body[k]

    return model, messages, openai_kwargs, bool(body.get("stream"))


def _chat_payload_to_response(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """chat.completion dict → Responses API ``response`` object (non-streaming)."""
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    usage = payload.get("usage") or {}
    msg_id = f"msg_{uuid.uuid4().hex}"
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": payload.get("created") or int(time.time()),
        "model": payload.get("model") or model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _responses_event(seq: int, event_type: str, payload: Dict[str, Any]) -> str:
    """Render one Responses API SSE event (both ``event:`` and ``data:`` lines)."""
    data = {"type": event_type, "sequence_number": seq, **payload}
    return f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"


async def _responses_sse_stream(
    model: str, messages: List[Dict[str, Any]], openai_kwargs: Dict[str, Any]
):
    """Bridge a chat-completion stream into the Responses API ``response.*``
    SSE event sequence (created → output_item/content_part added →
    output_text.delta* → *.done → completed)."""
    rid = f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    seq = 0

    def base(status: str, output: List[Dict[str, Any]], usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r: Dict[str, Any] = {
            "id": rid, "object": "response", "created_at": created,
            "model": model, "status": status, "output": output,
        }
        if usage is not None:
            r["usage"] = usage
        return r

    yield _responses_event(seq, "response.created", {"response": base("in_progress", [])}); seq += 1
    item_stub = {"id": msg_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
    yield _responses_event(seq, "response.output_item.added", {"output_index": 0, "item": item_stub}); seq += 1
    yield _responses_event(seq, "response.content_part.added", {
        "item_id": msg_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    }); seq += 1

    parts: List[str] = []
    usage_out = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async with _get_semaphore():
        try:
            async for chunk in _adapter.chat_completion_stream_async(
                model=model, messages=messages, **openai_kwargs
            ):
                cd = chunk.model_dump(exclude_none=True)
                ch = (cd.get("choices") or [{}])[0]
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    parts.append(delta)
                    yield _responses_event(seq, "response.output_text.delta", {
                        "item_id": msg_id, "output_index": 0, "content_index": 0, "delta": delta,
                    }); seq += 1
                u = cd.get("usage")
                if u:
                    usage_out = {
                        "input_tokens": u.get("prompt_tokens", 0),
                        "output_tokens": u.get("completion_tokens", 0),
                        "total_tokens": u.get("total_tokens", 0),
                    }
        except PaymentError as exc:
            yield _responses_event(seq, "response.failed", {
                "response": base("failed", []),
                "error": {"code": "payment_error", "message": _payment_error_sse_message(exc)},
            })
            return
        except APIError as exc:
            yield _responses_event(seq, "response.failed", {
                "response": base("failed", []),
                "error": {"code": "upstream_error", "message": str(exc)},
            })
            return
        except Exception as exc:  # noqa: BLE001
            if _is_solana_rpc_exc(exc):
                log.warning("solana rpc error during responses stream: %s", _solana_rpc_msg(exc))
                msg = _solana_rpc_msg(exc)
            else:
                log.exception("responses stream error")
                msg = str(exc)
            yield _responses_event(seq, "response.failed", {
                "response": base("failed", []),
                "error": {"code": "server_error", "message": msg},
            })
            return

    text = "".join(parts)
    yield _responses_event(seq, "response.output_text.done", {
        "item_id": msg_id, "output_index": 0, "content_index": 0, "text": text,
    }); seq += 1
    yield _responses_event(seq, "response.content_part.done", {
        "item_id": msg_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": text, "annotations": []},
    }); seq += 1
    final_item = {
        "id": msg_id, "type": "message", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    yield _responses_event(seq, "response.output_item.done", {"output_index": 0, "item": final_item}); seq += 1
    yield _responses_event(seq, "response.completed", {"response": base("completed", [final_item], usage_out)})


@app.post("/v1/responses", dependencies=[Depends(_require_token)])
async def responses(request: Request) -> Any:
    """OpenAI Responses API bridge → BlockRun Chat Completions."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    model, messages, openai_kwargs, stream = _responses_to_chat(body)
    if not model or not messages:
        raise HTTPException(400, "`model` and `input` are required")

    if stream:
        return StreamingResponse(
            _responses_sse_stream(model, messages, openai_kwargs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    async with _get_semaphore():
        try:
            payload = await _adapter.chat_completion_async(
                model=model, messages=messages, **openai_kwargs
            )
        except PaymentError as exc:
            return JSONResponse(status_code=402, content=_payment_error_payload(exc))
        except APIError as exc:
            status = exc.status_code if 400 <= getattr(exc, "status_code", 0) < 600 else 502
            return JSONResponse(status_code=status, content={"error": str(exc)})
        except Exception as exc:
            if _is_solana_rpc_exc(exc):
                log.warning("solana rpc error: %s", _solana_rpc_msg(exc))
                raise HTTPException(503, _solana_rpc_msg(exc))
            raise

    return _chat_payload_to_response(payload, model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="blockrun-litellm-proxy",
        description="OpenAI-compatible local proxy for BlockRun (x402 gateway).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4001, help="Bind port (default: 4001)")
    parser.add_argument(
        "--api-url",
        default=None,
        help="Override BlockRun API URL (default: https://blockrun.ai/api)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args()

    if args.api_url:
        os.environ["BLOCKRUN_API_URL"] = args.api_url

    # Fail fast if no wallet — better than waiting for first request.
    try:
        _adapter.get_sync_client()
    except ValueError as exc:
        parser.exit(2, f"\nWallet not configured:\n  {exc}\n")

    import uvicorn

    uvicorn.run(
        "blockrun_litellm.proxy:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
