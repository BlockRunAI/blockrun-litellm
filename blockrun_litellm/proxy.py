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
from typing import Any, Dict, List, Optional

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
            error_event = _openai_error_event(str(exc), code=402)
        except APIError as exc:
            error_event = _openai_error_event(str(exc), code=getattr(exc, "status_code", 500) or 500)
        except Exception as exc:
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
        yield _openai_error_event(str(exc), code=402)
    except APIError as exc:
        yield _openai_error_event(str(exc), code=getattr(exc, "status_code", 500) or 500)
    except Exception as exc:
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
            raise HTTPException(402, str(exc))
        except APIError as exc:
            status = exc.status_code if 400 <= getattr(exc, "status_code", 0) < 600 else 502
            return JSONResponse(status_code=status, content={"error": str(exc)})

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
            raise HTTPException(402, str(exc))
        except APIError as exc:
            status = exc.status_code if 400 <= getattr(exc, "status_code", 0) < 600 else 502
            return JSONResponse(status_code=status, content={"error": str(exc)})

    return result


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
