"""Tests for the native Anthropic /v1/messages passthrough on the sidecar.

Regression guards for three bugs found in review:

  #2  A streaming upstream 4xx/5xx must reach the client with its REAL status —
      not as an unconditional HTTP 200 text/event-stream the Anthropic SDK would
      mis-parse or hang on. (_forward_passthrough opens the upstream first, then
      returns a real error Response when status >= 400.)

  #3  /v1/messages and /v1/messages/count_tokens must acquire _get_semaphore(),
      like every other paid route, so the agentic Anthropic path can't stampede
      the gateway past the global concurrency cap.

  Plus: the streaming happy path still streams the body, and non-stream requests
  pass the upstream status/body through verbatim.
"""

from __future__ import annotations

import contextlib

import httpx
import pytest
from fastapi.testclient import TestClient

import blockrun_litellm.proxy as proxy


class _BytesStream(httpx.SyncByteStream):
    """An unconsumed sync stream so iter_raw() works — mirrors what
    client.send(stream=True) returns in production (content=... would mark the
    stream already-consumed and is only valid for the buffered/error paths)."""

    def __init__(self, data: bytes):
        self._data = data

    def __iter__(self):
        yield self._data

    def close(self) -> None:
        pass


class _FakeClient:
    """Stands in for the cached httpx.Client whose transport signs x402.

    Records the requests it is handed and replays a canned response. Real
    httpx.Response objects back the fakes so iter_raw()/read()/close() behave
    exactly as in production.
    """

    def __init__(self, response: httpx.Response):
        self._response = response
        self.stream_sends = 0
        self.posts = 0

    def build_request(self, method, url, content=None, headers=None):
        return httpx.Request(method, url, content=content, headers=headers)

    def send(self, request, stream=False):
        self.stream_sends += 1
        return self._response

    def post(self, url, content=None, headers=None):
        self.posts += 1
        return self._response


@pytest.fixture
def client() -> TestClient:
    return TestClient(proxy.app)


@pytest.fixture
def count_semaphore(monkeypatch: pytest.MonkeyPatch):
    """Replace _get_semaphore with a spy that counts acquisitions."""
    calls = {"acquired": 0}

    @contextlib.asynccontextmanager
    async def _spy():
        calls["acquired"] += 1
        yield

    monkeypatch.setattr(proxy, "_get_semaphore", _spy)
    return calls


def _patch_client(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> _FakeClient:
    fake = _FakeClient(response)
    monkeypatch.setattr(proxy, "_messages_client", lambda api_url: fake)
    return fake


# ---------------------------------------------------------------------------
# #2 — streaming error status is preserved, not masked as 200
# ---------------------------------------------------------------------------

def test_streaming_upstream_error_preserves_status(monkeypatch, client, count_semaphore):
    err = httpx.Response(
        400,
        headers={"content-type": "application/json"},
        content=b'{"type":"error","error":{"type":"invalid_request_error"}}',
    )
    _patch_client(monkeypatch, err)

    resp = client.post("/v1/messages", json={"model": "claude-haiku-4-5", "stream": True})

    # The real 400 reaches the client — NOT a 200 text/event-stream.
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    assert b"invalid_request_error" in resp.content
    assert count_semaphore["acquired"] == 1  # paid upstream was gated


def test_streaming_success_streams_body(monkeypatch, client, count_semaphore):
    ok = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_BytesStream(b"event: message_start\ndata: {}\n\n"),
    )
    _patch_client(monkeypatch, ok)

    resp = client.post("/v1/messages", json={"model": "claude-haiku-4-5", "stream": True})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert b"message_start" in resp.content
    assert count_semaphore["acquired"] == 1


# ---------------------------------------------------------------------------
# Non-stream passthrough preserves status + body verbatim
# ---------------------------------------------------------------------------

def test_non_stream_passthrough(monkeypatch, client, count_semaphore):
    ok = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=b'{"id":"msg_1","type":"message"}',
    )
    fake = _patch_client(monkeypatch, ok)

    resp = client.post("/v1/messages", json={"model": "claude-haiku-4-5"})

    assert resp.status_code == 200
    assert resp.content == b'{"id":"msg_1","type":"message"}'
    assert fake.posts == 1 and fake.stream_sends == 0  # took the buffered path
    assert count_semaphore["acquired"] == 1


def test_non_stream_upstream_error_passthrough(monkeypatch, client, count_semaphore):
    err = httpx.Response(402, headers={"content-type": "application/json"}, content=b'{"error":"pay"}')
    _patch_client(monkeypatch, err)

    resp = client.post("/v1/messages", json={"model": "claude-haiku-4-5"})

    assert resp.status_code == 402
    assert b"pay" in resp.content


# ---------------------------------------------------------------------------
# #3 — count_tokens is gated by the semaphore too
# ---------------------------------------------------------------------------

def test_count_tokens_acquires_semaphore(monkeypatch, client, count_semaphore):
    ok = httpx.Response(200, headers={"content-type": "application/json"}, content=b'{"input_tokens":42}')
    fake = _patch_client(monkeypatch, ok)

    resp = client.post("/v1/messages/count_tokens", json={"model": "claude-haiku-4-5", "messages": []})

    assert resp.status_code == 200
    assert b"42" in resp.content
    assert fake.posts == 1  # count_tokens is never streamed
    assert count_semaphore["acquired"] == 1


# ---------------------------------------------------------------------------
# /v1/chat/completions rides the SAME passthrough (PR #7) — verify it is wired
# to _forward_passthrough with allow_stream=True, not the old typed SDK path.
# ---------------------------------------------------------------------------

def test_chat_completions_non_stream_passthrough(monkeypatch, client, count_semaphore):
    ok = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=b'{"id":"chatcmpl_1","object":"chat.completion"}',
    )
    fake = _patch_client(monkeypatch, ok)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    assert resp.content == b'{"id":"chatcmpl_1","object":"chat.completion"}'
    assert fake.posts == 1 and fake.stream_sends == 0  # buffered (non-stream) path
    assert count_semaphore["acquired"] == 1


def test_chat_completions_streaming_error_preserves_status(monkeypatch, client, count_semaphore):
    err = httpx.Response(
        402,
        headers={"content-type": "application/json"},
        content=b'{"error":{"message":"insufficient funds"}}',
    )
    fake = _patch_client(monkeypatch, err)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-5.5", "messages": [], "stream": True},
    )

    # The real 402 reaches the OpenAI client — NOT a 200 SSE that masks the error.
    assert resp.status_code == 402
    assert b"insufficient funds" in resp.content
    assert fake.stream_sends == 1  # streaming path opened the upstream first
    assert count_semaphore["acquired"] == 1


# ---------------------------------------------------------------------------
# A Solana RPC fault during x402 signing maps to 503, not a bare 500 (PR #7).
# Applies to every passthrough route; exercised here via /v1/chat/completions.
# ---------------------------------------------------------------------------

class _RaisingClient:
    """A signing-stage failure: post()/send() raise before any upstream status."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def build_request(self, method, url, content=None, headers=None):
        return httpx.Request(method, url, content=content, headers=headers)

    def send(self, request, stream=False):
        raise self._exc

    def post(self, url, content=None, headers=None):
        raise self._exc


def test_signing_solana_rpc_error_maps_to_503(monkeypatch, client, count_semaphore):
    boom = RuntimeError("getAccountInfo timed out")
    monkeypatch.setattr(proxy, "_messages_client", lambda api_url: _RaisingClient(boom))
    # Treat our sentinel as a Solana RPC fault without constructing the real
    # (optional-dep) SolanaRpcException, whose ctor needs a live RPC context.
    monkeypatch.setattr(proxy, "_is_solana_rpc_exc", lambda exc: exc is boom)
    monkeypatch.setattr(proxy, "_solana_rpc_msg", lambda exc: str(exc))

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 503
    assert b"getAccountInfo timed out" in resp.content
    assert count_semaphore["acquired"] == 1


def test_signing_non_solana_error_propagates(monkeypatch, count_semaphore):
    # A non-Solana signing fault must NOT be silently downgraded to 503 — it
    # propagates (FastAPI 500), same as the old typed chat path re-raised.
    boom = RuntimeError("unexpected")
    monkeypatch.setattr(proxy, "_messages_client", lambda api_url: _RaisingClient(boom))
    monkeypatch.setattr(proxy, "_is_solana_rpc_exc", lambda exc: False)

    # raise_server_exceptions=True (default) surfaces the original exception so we
    # can assert it is NOT swallowed into a 503.
    strict = TestClient(proxy.app)
    with pytest.raises(RuntimeError, match="unexpected"):
        strict.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-5.5", "messages": [], "stream": True},
        )
