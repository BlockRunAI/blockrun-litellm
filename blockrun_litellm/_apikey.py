"""
The API-key rail — a BlockRun account instead of an x402 wallet.

Why this exists
---------------
Everything else in this package pays per call from a wallet the caller holds:
the ``blockrun-llm`` SDK signs an x402 payment (EIP-712 on Base, SVM on
Solana) and the gateway settles it on-chain. That is the agent-native path and
it stays the default for anyone who has a wallet.

It is also a wall for most people who just want to try the thing. An API key
removes the wallet entirely: you sign in at ``user.blockrun.ai``, top up with a
card, and get a ``brk_live_…`` key that authenticates against
``api.blockrun.ai`` — an OpenAI-shaped surface that fronts the same gateway and
the same model catalogue. BlockRun pays the gateway on your behalf and debits
your prepaid credit at the published list price.

What this module is
-------------------
A small HTTP transport with the *same return shapes* as the SDK calls it
replaces, so nothing above ``_adapter`` has to know which rail served a
request:

* chat / responses / media  → the gateway's JSON, parsed to a ``dict``
  (identical to ``response.model_dump(exclude_none=True)`` on the SDK path,
  because ``api.blockrun.ai`` forwards the gateway's body verbatim)
* streaming chat            → ``ChatCompletionChunk`` objects, the exact type
  ``LLMClient.chat_completion_stream`` yields

Deliberately NOT here
---------------------
* **Chain selection.** There is no chain on this rail. Credit is prepaid off
  chain, so there is no Base-vs-Solana choice to make and no wallet to fund.
  ``BLOCKRUN_CHAIN`` / ``BLOCKRUN_API_URL`` are ignored when a key is in play.
* **Real per-call cost.** The wallet rail reports the exact on-chain charge
  from the x402 settlement header. ``api.blockrun.ai`` returns no such header —
  it did not make an on-chain payment for this call — so ``cost_usd`` stays
  ``None`` and LiteLLM's token x list-price estimate is what gets logged.
  Authoritative spend lives in the account ledger at ``user.blockrun.ai``.
  Reporting an estimate as if it were a settled charge would be worse than
  reporting nothing, which is why nothing is what this returns.

Endpoint coverage
-----------------
Chat (incl. streaming), the Anthropic Messages dialect, Responses, images,
video (submit + poll + download), speech, music and sound effects all work.
The native Gemini surface (``/v1beta/models/…``) does **not** — see
:data:`UNSUPPORTED_PATHS`. Callers get a 501 naming the wallet rail as the way
to reach it, rather than a bare 404 from a host that never published it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple

import httpx
from blockrun_llm.types import APIError, ChatCompletionChunk, PaymentError

_log = logging.getLogger(__name__)


# The account API. Unlike the gateway URLs this has no ``/api`` path segment:
# ``api.blockrun.ai/v1/chat/completions``, not ``.../api/v1/...``.
DEFAULT_API_BASE = "https://api.blockrun.ai"

# Every BlockRun-issued key starts with this. It is what lets one ``api_key``
# argument carry either credential without a second parameter: a wallet private
# key is hex (Base) or base58 (Solana) and can never collide with it.
KEY_PREFIX = "brk_"

# Where a caller goes to get a key and add credit. Quoted in errors, because an
# authentication failure that does not say where to get a working credential is
# a dead end.
PORTAL_URL = "https://user.blockrun.ai"

# Paths the account API does not publish. The value is what the caller is told.
UNSUPPORTED_PATHS: Dict[str, str] = {
    "/v1beta": (
        "The native Gemini protocol (/v1beta/models/...) is served only on the "
        "x402 wallet rail. Unset BLOCKRUN_API_KEY and configure a wallet "
        "(SOLANA_WALLET_KEY or BLOCKRUN_WALLET_KEY) to use it, or call Gemini "
        "through /v1/chat/completions with model=google/gemini-3-pro."
    ),
}

# ``/v1/images/edits`` is the OpenAI spelling and is what this proxy exposes;
# the gateway publishes the same operation as ``image2image`` and the account
# API allowlists it under that name only. Rewriting here keeps the OpenAI alias
# working on both rails instead of 404ing on one of them.
_PATH_REWRITES: Dict[str, str] = {
    "/v1/images/edits": "/v1/images/image2image",
}


def looks_like_api_key(value: Optional[str]) -> bool:
    """True when a credential is a BlockRun API key rather than a wallet key.

    Used to let a single ``api_key`` argument — LiteLLM's, or the SDK's
    ``private_key`` — carry either one. The prefix is issued by us and no
    private key format begins with it, so the test cannot misfire.
    """
    return isinstance(value, str) and value.strip().startswith(KEY_PREFIX)


def resolve_api_key(*candidates: Optional[str]) -> Optional[str]:
    """First BlockRun API key among the candidates, else ``BLOCKRUN_API_KEY``.

    Candidates are the per-call credentials in precedence order. A candidate
    that is not a BlockRun key is skipped rather than rejected: it is a wallet
    key, which is a legitimate value for the same argument.
    """
    for candidate in candidates:
        if looks_like_api_key(candidate):
            return candidate.strip()  # type: ignore[union-attr]
    env = os.environ.get("BLOCKRUN_API_KEY", "").strip()
    return env or None


def api_base() -> str:
    """Account API base URL. ``BLOCKRUN_API_BASE_URL`` overrides for staging."""
    return (os.environ.get("BLOCKRUN_API_BASE_URL") or DEFAULT_API_BASE).rstrip("/")


def unsupported_reason(path: str) -> Optional[str]:
    """Why ``path`` cannot be served on this rail, or ``None`` if it can."""
    for prefix, reason in UNSUPPORTED_PATHS.items():
        if path.startswith(prefix):
            return reason
    return None


def target_url(path: str) -> str:
    """Absolute account-API URL for a proxy-relative path."""
    return f"{api_base()}{_PATH_REWRITES.get(path, path)}"


def headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Shared HTTP clients
# ---------------------------------------------------------------------------
# One pooled client per (sync|async) process, not per call: a fresh
# httpx.Client per request re-runs TLS on every completion and, under the
# proxy's 100-way concurrency, leaks sockets faster than they close. Limits
# mirror the wallet rail's pool, minus the doubling — this rail sends one
# request per call, not a 402 probe plus a signed retry.

_HTTP_LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=50)

_sync_http: Optional[httpx.Client] = None
_async_http: Optional[httpx.AsyncClient] = None
_http_lock = threading.Lock()


def _timeout(seconds: Optional[float]) -> httpx.Timeout:
    """Long read, short connect.

    A reasoning model can think for minutes before the first byte, so the read
    budget has to be generous; a host that is not answering at all should still
    fail in seconds rather than inherit that budget.
    """
    return httpx.Timeout(seconds or 600.0, connect=15.0)


def sync_http() -> httpx.Client:
    global _sync_http
    with _http_lock:
        if _sync_http is None:
            _sync_http = httpx.Client(limits=_HTTP_LIMITS, timeout=_timeout(None))
        return _sync_http


def async_http() -> httpx.AsyncClient:
    global _async_http
    with _http_lock:
        if _async_http is None:
            _async_http = httpx.AsyncClient(limits=_HTTP_LIMITS, timeout=_timeout(None))
        return _async_http


def _reset_clients_for_tests() -> None:
    """Drop the pooled clients. Test-only; the process otherwise keeps them."""
    global _sync_http, _async_http
    with _http_lock:
        _sync_http = None
        _async_http = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _error_message(status: int, body: bytes) -> str:
    """Pull the human part out of an account-API error body.

    Three shapes reach here: the account API's own OpenAI-style
    ``{"error": {"message", "type", "code"}}``, the gateway's flatter
    ``{"error": "..."}`` forwarded verbatim, and — on an infrastructure fault —
    HTML from a load balancer. The last one must not be echoed as a "message",
    so anything unparseable degrades to the status line.
    """
    try:
        parsed = json.loads(body or b"{}")
    except ValueError:
        return f"BlockRun API returned HTTP {status}"
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, dict):
        message = err.get("message")
        code = err.get("code")
        if isinstance(message, str) and message:
            return f"{message} (code={code})" if code else message
    if isinstance(err, str) and err:
        return err
    return f"BlockRun API returned HTTP {status}"


def raise_for_status(status: int, body: bytes) -> None:
    """Translate an account-API failure into the SDK's exception vocabulary.

    The proxy and the provider already branch on ``PaymentError`` vs
    ``APIError``; mapping onto those two means neither has to learn a third.

    * 401/403 → ``APIError``, with the portal URL appended. An invalid key is
      not retriable and the message has to say what to do about it.
    * 402     → ``PaymentError``. Same class the wallet rail raises when it
      cannot pay, so the 402 the proxy already returns keeps working — only
      the remedy differs (add credit, not fund a wallet).
    """
    if status < 400:
        return
    message = _error_message(status, body)
    if status == 402:
        raise PaymentError(
            f"{message}. Add credit at {PORTAL_URL}/dashboard/billing.",
            status_code=status,
            response={"raw": message},
        )
    if status in (401, 403):
        raise APIError(
            f"{message}. Check BLOCKRUN_API_KEY, or issue a new key at "
            f"{PORTAL_URL}/dashboard/keys.",
            status,
        )
    raise APIError(message, status)


# ---------------------------------------------------------------------------
# JSON calls
# ---------------------------------------------------------------------------


def post_json(
    path: str,
    body: Dict[str, Any],
    *,
    api_key: str,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """POST JSON, return the parsed response. Raises on non-2xx."""
    resp = sync_http().post(
        target_url(path), json=body, headers=headers(api_key), timeout=_timeout(timeout)
    )
    raise_for_status(resp.status_code, resp.content)
    return resp.json()


async def apost_json(
    path: str,
    body: Dict[str, Any],
    *,
    api_key: str,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Async :func:`post_json`."""
    resp = await async_http().post(
        target_url(path), json=body, headers=headers(api_key), timeout=_timeout(timeout)
    )
    raise_for_status(resp.status_code, resp.content)
    return resp.json()


def get_json(path: str, *, api_key: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    resp = sync_http().get(
        target_url(path), headers=headers(api_key), timeout=_timeout(timeout)
    )
    raise_for_status(resp.status_code, resp.content)
    return resp.json()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _sse_payloads(line: str) -> Optional[Dict[str, Any]]:
    """One SSE ``data:`` line to a chunk dict, or ``None`` to skip it.

    ``[DONE]``, comments and blank lines are structure, not data. Malformed
    JSON is dropped with a warning rather than killing the stream: a single bad
    frame must not lose the completion the caller has already paid for.
    """
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except ValueError:
        _log.warning("dropping unparseable SSE frame from the BlockRun API")
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_chunk(parsed: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
    """Validate a chunk dict into the SDK's type, keeping native extras.

    ``ChatCompletionChunk`` is configured ``extra="allow"``, so
    ``system_fingerprint`` / ``service_tier`` / ``*_tokens_details`` survive —
    which is the whole point of the gateway forwarding the upstream body
    verbatim. A frame that fails validation is skipped for the same reason a
    malformed one is: partial output beats no output.
    """
    try:
        return ChatCompletionChunk.model_validate(parsed)
    except Exception:  # noqa: BLE001 - pydantic raises its own error type
        _log.warning("dropping non-conforming chat chunk from the BlockRun API")
        return None


def stream_chat(
    body: Dict[str, Any],
    *,
    api_key: str,
    timeout: Optional[float] = None,
) -> Iterator[ChatCompletionChunk]:
    """Stream ``/v1/chat/completions``, yielding SDK-typed chunks.

    The upstream status is checked *before* any frame is yielded — a 402 or a
    429 arrives as an ordinary JSON error body with a non-200 status, and
    turning that into an empty stream would hide it from the caller.
    """
    payload = dict(body)
    payload["stream"] = True
    with sync_http().stream(
        "POST",
        target_url("/v1/chat/completions"),
        json=payload,
        headers=headers(api_key, {"Accept": "text/event-stream"}),
        timeout=_timeout(timeout),
    ) as resp:
        if resp.status_code >= 400:
            raise_for_status(resp.status_code, resp.read())
        for line in resp.iter_lines():
            parsed = _sse_payloads(line)
            if parsed is None:
                continue
            chunk = _as_chunk(parsed)
            if chunk is not None:
                yield chunk


async def astream_chat(
    body: Dict[str, Any],
    *,
    api_key: str,
    timeout: Optional[float] = None,
) -> AsyncIterator[ChatCompletionChunk]:
    """Async :func:`stream_chat`."""
    payload = dict(body)
    payload["stream"] = True
    async with async_http().stream(
        "POST",
        target_url("/v1/chat/completions"),
        json=payload,
        headers=headers(api_key, {"Accept": "text/event-stream"}),
        timeout=_timeout(timeout),
    ) as resp:
        if resp.status_code >= 400:
            raise_for_status(resp.status_code, await resp.aread())
        async for line in resp.aiter_lines():
            parsed = _sse_payloads(line)
            if parsed is None:
                continue
            chunk = _as_chunk(parsed)
            if chunk is not None:
                yield chunk


# ---------------------------------------------------------------------------
# Video: submit + poll
# ---------------------------------------------------------------------------
# Video is the one medium that is not a single request. The gateway answers a
# submit with a job id and generates asynchronously, so the SDK's VideoClient
# submits then polls to completion, and its callers — including this package's
# proxy, which layers the OpenAI Videos job API on top — expect a *finished*
# video back. This reproduces that loop, minus the x402 half: no 402 probe, no
# signature to re-sign when the 600s authorization window lapses. Statuses and
# the give-up behaviour deliberately match the SDK's so the two rails fail the
# same way.

_POLL_INTERVAL_S = 5.0
_DEFAULT_VIDEO_BUDGET_S = 900.0


def submit_and_poll_video(
    body: Dict[str, Any],
    *,
    api_key: str,
    budget_seconds: Optional[float] = None,
    sleep: Any = None,
) -> Dict[str, Any]:
    """Submit a video job and poll until it finishes. Returns the final job.

    ``sleep`` is injectable so tests do not spend real seconds waiting.
    """
    nap = sleep or time.sleep
    submitted = post_json("/v1/videos/generations", body, api_key=api_key, timeout=120.0)
    job_id = submitted.get("id")
    if not job_id:
        raise APIError("Video submit response carried no job id", 502, {"response": submitted})

    status = submitted.get("status", "queued")
    if status == "completed":
        return submitted

    deadline = time.monotonic() + (budget_seconds or _DEFAULT_VIDEO_BUDGET_S)
    while time.monotonic() < deadline:
        nap(_POLL_INTERVAL_S)
        resp = sync_http().get(
            target_url(f"/v1/videos/{job_id}"), headers=headers(api_key), timeout=_timeout(60.0)
        )
        try:
            poll = resp.json()
        except ValueError:
            poll = {}
        status = poll.get("status", status)
        if status == "completed":
            return poll
        if status == "failed":
            raise APIError(
                f"Upstream generation failed: {poll.get('error', 'unknown')}",
                resp.status_code,
                poll,
            )
        # 504 is a transient upstream hiccup on the gateway's own poll; 202 is
        # "still working". Anything else is a real failure and is raised.
        if resp.status_code not in (200, 202, 504):
            raise_for_status(resp.status_code, resp.content)

    raise APIError(
        f"Video generation did not complete within the {budget_seconds or _DEFAULT_VIDEO_BUDGET_S:.0f}s "
        f"budget (last status: {status}). The job is not lost — it stays claimable for "
        f"~48h; GET /v1/videos/{job_id} to fetch it once it finishes.",
        504,
        {"id": job_id, "last_status": status},
    )


# ---------------------------------------------------------------------------
# Raw passthrough (used by the proxy for /v1/messages and friends)
# ---------------------------------------------------------------------------


def passthrough_headers(inbound: Dict[str, str], api_key: str) -> Dict[str, str]:
    """Headers for a byte-for-byte forward.

    The client's own ``Authorization`` is *replaced*, never forwarded: on this
    rail that header carried the proxy's optional shared secret
    (``BLOCKRUN_PROXY_TOKEN``), and leaking a local secret to a remote host
    would be a worse bug than the one it guards against.
    """
    out = {k: v for k, v in inbound.items() if k.lower() not in ("authorization", "host")}
    out["Authorization"] = f"Bearer {api_key}"
    return out


def open_stream(
    method: str,
    path: str,
    content: bytes,
    hdrs: Dict[str, str],
    *,
    timeout: Optional[float] = None,
) -> Tuple[httpx.Response, Any]:
    """Open an un-read streaming response, mirroring the wallet rail's helper.

    Returns ``(response, context)``; the caller must close the context. Split
    this way so the caller can inspect ``status_code`` before deciding whether
    to stream a body or surface an error.
    """
    ctx = sync_http().stream(
        method, target_url(path), content=content, headers=hdrs, timeout=_timeout(timeout)
    )
    return ctx.__enter__(), ctx


__all__ = [
    "DEFAULT_API_BASE",
    "KEY_PREFIX",
    "PORTAL_URL",
    "UNSUPPORTED_PATHS",
    "api_base",
    "apost_json",
    "astream_chat",
    "async_http",
    "get_json",
    "headers",
    "looks_like_api_key",
    "open_stream",
    "passthrough_headers",
    "post_json",
    "raise_for_status",
    "resolve_api_key",
    "stream_chat",
    "submit_and_poll_video",
    "sync_http",
    "target_url",
    "unsupported_reason",
]
