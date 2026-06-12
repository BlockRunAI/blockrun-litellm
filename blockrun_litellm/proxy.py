"""
Local OpenAI-compatible proxy for BlockRun.

Run as a sidecar; point any OpenAI client (LiteLLM, langchain, raw SDK,
curl) at ``http://localhost:4001/v1`` and it Just Works. Your x402 wallet
key lives in this process — clients on the same host never see it.

Endpoints
---------
- ``POST /v1/chat/completions``      — OpenAI Chat Completions
- ``POST /v1/messages``              — native Anthropic Messages (Claude Code,
  Anthropic SDK); verbatim x402-signed passthrough — tools/thinking preserved
- ``POST /v1/messages/count_tokens`` — Anthropic token counting (passthrough)
- ``POST /v1/images/generations``    — OpenAI Image Generations (DALL-E compatible)
- ``GET  /v1/models``                — passthrough to BlockRun's chat catalog
- ``GET  /healthz``                  — liveness probe (no upstream call)

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
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse, Response, StreamingResponse
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "blockrun-litellm[proxy] extras not installed. "
        "Run: pip install 'blockrun-litellm[proxy]'"
    ) from exc

import httpx
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


@app.post("/v1/chat/completions", dependencies=[Depends(_require_token)])
async def chat_completions(request: Request) -> Any:
    # Verbatim x402-signed passthrough to the gateway's native
    # /v1/chat/completions. Previously this went through the SDK's typed
    # chat_completion_stream, which crashes on streamed tool calls
    # ('dict' object has no attribute 'delta' — the strict ToolCall schema
    # rejects streaming argument-fragment frames, falls back to model_construct,
    # then the archive loop reads .delta on a dict). Raw passthrough keeps every
    # OpenAI client — including Codex with wire_api=chat — off that path, so
    # streamed tool_calls survive intact.
    return await _forward_passthrough(
        request, "/v1/chat/completions", _openai_fwd_headers(request), allow_stream=True
    )


# ---------------------------------------------------------------------------
# Native Anthropic /v1/messages passthrough (Claude Code, Anthropic SDK, ...)
# ---------------------------------------------------------------------------
# Claude Code (and anything built on the Anthropic SDK) speaks the Anthropic
# Messages protocol, NOT OpenAI. Rather than translate Anthropic<->OpenAI
# (lossy — silently drops tool_use / thinking / cache_control), we forward the
# request VERBATIM to BlockRun's native /v1/messages endpoint and add ONLY the
# x402 signature. tools / tool_choice / thinking / streaming pass through
# untouched, so agentic tool use works exactly as against api.anthropic.com.
#
# Both chains supported: Base signs via the SDK's EIP-712 httpx transport, Solana
# via _SolanaX402Transport (SVM). The chain is picked from the configured
# api-url (--api-url / BLOCKRUN_API_URL).

_messages_http_clients: Dict[str, httpx.Client] = {}


class _SolanaX402Transport(httpx.BaseTransport):
    """httpx transport that signs Solana (SVM) x402 payments on 402 — the Solana
    twin of blockrun_llm's Base/EIP-712 ``_BlockRunX402Transport``.

    Lets ANY path (here: ``/v1/messages``) stream through ``sol.blockrun.ai`` by
    reusing the SDK's SVM signing helpers. The x402 client isn't thread-safe, so
    the brief signing step is lock-guarded (mirrors SolanaLLMClient._sign_payment).
    """

    def __init__(self, private_key: str, api_url: str, base_transport=None):
        # Lazy import: the x402 SVM stack only ships with blockrun-llm[solana].
        from x402 import x402ClientSync
        from blockrun_llm.solana_client import (
            _create_signer,
            _resolve_rpc_config,
            _register_svm_with_headers,
        )

        self._api_url = api_url.rstrip("/")
        self._base = base_transport or httpx.HTTPTransport()
        resolved_url, resolved_headers = _resolve_rpc_config(None, None)
        self._x402_client = x402ClientSync()
        _register_svm_with_headers(
            self._x402_client, _create_signer(private_key), resolved_url, resolved_headers
        )
        self._lock = threading.Lock()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        from blockrun_llm.solana_client import SolanaLLMClient
        from x402.http.utils import (
            decode_payment_required_header,
            encode_payment_signature_header,
        )

        response = self._base.handle_request(request)
        if response.status_code != 402:
            return response

        response.read()
        payment_header = SolanaLLMClient._extract_payment_header(response)
        if not payment_header:
            return response

        payment_required = decode_payment_required_header(payment_header)
        with self._lock:
            payload = self._x402_client.create_payment_payload(payment_required)
        request.headers["PAYMENT-SIGNATURE"] = encode_payment_signature_header(payload)
        return self._base.handle_request(request)

    def close(self) -> None:
        self._base.close()


def _resolve_api_url() -> str:
    return (os.environ.get("BLOCKRUN_API_URL") or _adapter.BASE_API_URL).rstrip("/")


def _messages_client(api_url: str) -> httpx.Client:
    """Cached httpx client whose transport signs x402 for any path on the chain
    implied by ``api_url`` (Base via EIP-712, Solana via SVM)."""
    existing = _messages_http_clients.get(api_url)
    if existing is not None:
        return existing

    if _adapter._is_solana_url(api_url):
        from blockrun_llm.solana_wallet import load_solana_wallet

        key = os.environ.get("SOLANA_WALLET_KEY") or load_solana_wallet()
        if not key:
            raise HTTPException(500, "No Solana wallet configured (set SOLANA_WALLET_KEY)")
        transport: httpx.BaseTransport = _SolanaX402Transport(private_key=key, api_url=api_url)
    else:
        from eth_account import Account
        from blockrun_llm.anthropic_client import _BlockRunX402Transport
        from blockrun_llm.wallet import load_wallet

        key = os.environ.get("BLOCKRUN_WALLET_KEY") or load_wallet()
        if not key:
            raise HTTPException(500, "No Base wallet configured (set BLOCKRUN_WALLET_KEY)")
        if not key.startswith("0x"):
            key = "0x" + key
        transport = _BlockRunX402Transport(account=Account.from_key(key), api_url=api_url)

    client = httpx.Client(transport=transport, timeout=300.0)
    _messages_http_clients[api_url] = client
    return client


def _anthropic_fwd_headers(request: Request) -> Dict[str, str]:
    # Strip the client's proxy-gate auth (Authorization / x-api-key); the
    # transport supplies its own x402 PAYMENT-SIGNATURE. Mirror AnthropicClient,
    # which sets api_key="blockrun". Preserve the Anthropic protocol headers.
    headers = {
        "content-type": "application/json",
        "x-api-key": "blockrun",
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }
    beta = request.headers.get("anthropic-beta")
    if beta:
        headers["anthropic-beta"] = beta
    return headers


def _openai_fwd_headers(request: Request) -> Dict[str, str]:
    # Drop the client's proxy-gate auth; the x402 PAYMENT-SIGNATURE comes from
    # the transport, and the gateway treats x402 as the auth (no API key needed).
    return {"content-type": request.headers.get("content-type", "application/json")}


def _open_upstream_stream(
    client: httpx.Client, target: str, raw: bytes, headers: Dict[str, str]
) -> httpx.Response:
    """Send a streaming POST and return the response with its status known but
    body unread. The caller owns the response and MUST close it. Opening the
    stream up front (rather than inside the response generator) is what lets us
    surface the real upstream status instead of an unconditional ``200``."""
    req = client.build_request("POST", target, content=raw, headers=headers)
    return client.send(req, stream=True)


async def _forward_passthrough(
    request: Request, path: str, headers: Dict[str, str], *, allow_stream: bool
) -> Response:
    """Verbatim x402-signed passthrough to a BlockRun native endpoint.

    Shared by ``/v1/messages``, ``/v1/messages/count_tokens`` and
    ``/v1/chat/completions``. The body is forwarded byte-for-byte — no
    translation, no SDK typed parsing — so OpenAI clients (incl. Codex with
    wire_api=chat) and Claude Code stay off the SDK's ``chat_completion_stream``
    path, the source of the streamed-tool-call ``'dict' object has no attribute
    'delta'`` crash. The upstream call is gated by ``_get_semaphore()`` like
    every other paid route. For streaming we open the upstream first to learn its
    real status, then either return a real error ``Response`` (preserving the
    4xx/5xx the client must see) or stream the body.
    """
    api_url = _resolve_api_url()
    raw = await request.body()
    client = _messages_client(api_url)
    qs = request.url.query
    target = f"{api_url}{path}" + (f"?{qs}" if qs else "")

    wants_stream = False
    if allow_stream:
        try:
            wants_stream = bool(_json.loads(raw or b"{}").get("stream"))
        except Exception:
            wants_stream = False

    if wants_stream:
        # Hold the semaphore across the paid upstream connection (x402 probe +
        # sign + handshake) — released before the body drain so a slow reader
        # reading at 1 tok/s doesn't keep a slot and block other callers.
        async with _get_semaphore():
            resp = await run_in_threadpool(_open_upstream_stream, client, target, raw, headers)

        if resp.status_code >= 400:
            # A real upstream error — surface the true status + body, NOT a 200
            # text/event-stream the Anthropic SDK would mis-parse or hang on.
            body = await run_in_threadpool(resp.read)
            await run_in_threadpool(resp.close)
            return Response(
                content=body,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

        def _drain():
            try:
                for chunk in resp.iter_raw():
                    yield chunk
            finally:
                resp.close()

        return StreamingResponse(
            _drain(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/event-stream"),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _post():
        resp = client.post(target, content=raw, headers=headers)
        return resp.status_code, resp.headers.get("content-type", "application/json"), resp.content

    async with _get_semaphore():
        status, ctype, content = await run_in_threadpool(_post)
    return Response(content=content, status_code=status, media_type=ctype)


@app.post("/v1/messages", dependencies=[Depends(_require_token)])
async def anthropic_messages(request: Request) -> Any:
    return await _forward_passthrough(
        request, "/v1/messages", _anthropic_fwd_headers(request), allow_stream=True
    )


@app.post("/v1/messages/count_tokens", dependencies=[Depends(_require_token)])
async def anthropic_count_tokens(request: Request) -> Any:
    # Claude Code calls this to size requests; count_tokens is never streamed.
    return await _forward_passthrough(
        request, "/v1/messages/count_tokens", _anthropic_fwd_headers(request), allow_stream=False
    )


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
