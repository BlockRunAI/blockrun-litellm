"""Tests for the native Anthropic /v1/messages passthrough on the sidecar.

Regression guards for three bugs found in review:

  #2  A streaming upstream 4xx/5xx must reach the client with its REAL status —
      not as an unconditional HTTP 200 text/event-stream the Anthropic SDK would
      mis-parse or hang on. (_forward_anthropic opens the upstream first, then
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
