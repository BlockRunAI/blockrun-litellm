"""Native Gemini protocol coverage for the x402 sidecar.

The sidecar must preserve Gemini JSON/SSE bytes while adding wallet payment.
Streaming is selected by ``:streamGenerateContent`` in the URL, not by a
``stream`` field in the request body.
"""

from __future__ import annotations

import contextlib

import httpx
import pytest
from fastapi.testclient import TestClient

import blockrun_litellm.proxy as proxy


class _BytesStream(httpx.SyncByteStream):
    def __init__(self, data: bytes):
        self._data = data

    def __iter__(self):
        yield self._data

    def close(self) -> None:
        pass


class _FakeClient:
    def __init__(self, response: httpx.Response):
        self._response = response
        self.requests: list[httpx.Request] = []
        self.posts = 0
        self.stream_sends = 0

    def build_request(self, method, url, content=None, headers=None):
        request = httpx.Request(method, url, content=content, headers=headers)
        self.requests.append(request)
        return request

    def send(self, request, stream=False):
        self.stream_sends += 1
        return self._response

    def post(self, url, content=None, headers=None):
        self.posts += 1
        request = httpx.Request("POST", url, content=content, headers=headers)
        self.requests.append(request)
        return self._response


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # _resolve_api_url() reads BLOCKRUN_API_URL first; a developer or CI job with
    # it exported (the documented Solana workflow) would otherwise flip the
    # absolute-URL assertions below. Pin the Base default for these tests.
    monkeypatch.delenv("BLOCKRUN_API_URL", raising=False)
    return TestClient(proxy.app)


@pytest.fixture
def count_semaphore(monkeypatch: pytest.MonkeyPatch):
    calls = {"acquired": 0}

    @contextlib.asynccontextmanager
    async def _spy():
        calls["acquired"] += 1
        yield

    monkeypatch.setattr(proxy, "_get_semaphore", _spy)
    return calls


def _patch_client(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> _FakeClient:
    fake = _FakeClient(response)
    monkeypatch.setattr(proxy, "_messages_client", lambda api_url: fake)
    return fake


def test_generate_content_is_verbatim_and_drops_google_credentials(
    monkeypatch, client, count_semaphore
):
    upstream_body = b'{"candidates":[{"content":{"parts":[{"text":"native-ok"}]}}]}'
    fake = _patch_client(
        monkeypatch,
        httpx.Response(200, headers={"content-type": "application/json"}, content=upstream_body),
    )
    raw = b'{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}'

    response = client.post(
        "/v1beta/models/gemini-2.5-flash:generateContent?key=client-secret",
        content=raw,
        headers={
            "content-type": "application/json",
            "x-goog-api-key": "client-secret",
            "authorization": "Bearer proxy-or-google-token",
        },
    )

    assert response.status_code == 200
    assert response.content == upstream_body
    assert fake.posts == 1 and fake.stream_sends == 0
    assert count_semaphore["acquired"] == 1
    forwarded = fake.requests[0]
    assert str(forwarded.url) == (
        "https://blockrun.ai/api/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert forwarded.content == raw
    assert "x-goog-api-key" not in forwarded.headers
    assert "authorization" not in forwarded.headers


def test_stream_generate_content_uses_url_method_and_preserves_sse(
    monkeypatch, client, count_semaphore
):
    sse = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"stream-ok"}]}}]}\n\n'
        b'data: {"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}}\n\n'
    )
    fake = _patch_client(
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_BytesStream(sse),
        ),
    )

    response = client.post(
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=dummy",
        json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == sse
    assert fake.stream_sends == 1 and fake.posts == 0
    assert count_semaphore["acquired"] == 1
    assert str(fake.requests[0].url) == (
        "https://blockrun.ai/api/v1beta/models/"
        "gemini-2.5-flash:streamGenerateContent"
    )


def test_streaming_upstream_error_keeps_native_status(monkeypatch, client, count_semaphore):
    error = b'{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}'
    fake = _patch_client(
        monkeypatch,
        httpx.Response(429, headers={"content-type": "application/json"}, content=error),
    )

    response = client.post(
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        json={"contents": []},
    )

    assert response.status_code == 429
    assert response.content == error
    assert fake.stream_sends == 1
    assert count_semaphore["acquired"] == 1


def test_google_prefixed_model_is_stripped_before_forwarding(monkeypatch, client, count_semaphore):
    """A ``google/`` (or ``models/``) prefix is an id-only convenience the gateway
    also strips. We rebuild the forwarded path from the sanitized model so the
    upstream sees the bare id, and the audit log canonicalizes to ``google/<id>``.
    """
    fake = _patch_client(
        monkeypatch,
        httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}"),
    )
    logged = []
    monkeypatch.setattr(proxy._logger, "log_proxy_call", lambda **kwargs: logged.append(kwargs))

    response = client.post(
        "/v1beta/models/google/gemini-3.1-pro:generateContent",
        json={"contents": []},
    )

    assert response.status_code == 200
    # Path is rebuilt from the sanitized model — the ``google/`` prefix is gone.
    assert str(fake.requests[0].url).endswith(
        "/v1beta/models/gemini-3.1-pro:generateContent"
    )
    assert logged[0]["model"] == "google/gemini-3.1-pro"
    assert logged[0]["path"] == "/v1beta/models/gemini-3.1-pro:generateContent"


@pytest.mark.parametrize(
    "path",
    [
        "/v1beta/models/gemini-2.5-flash:countTokens",
        "/v1beta/models/gemini-2.5-flash",
        # Blank / whitespace-only model — the model token is empty after stripping.
        "/v1beta/models/:generateContent",
        "/v1beta/models/%20:streamGenerateContent",
        # A slash in the model reaches the handler (the :path converter keeps it)
        # and must be rejected before payment — this is the escape the sanitizer
        # closes. httpx would otherwise normalize it out of /v1beta/models/.
        "/v1beta/models/foo/bar:generateContent",
        "/v1beta/models/google/gemini/escape:generateContent",
    ],
)
def test_unsupported_or_unsafe_native_method_fails_before_payment(monkeypatch, client, path):
    def _must_not_create_client(api_url):
        raise AssertionError("wallet client must not be created")

    monkeypatch.setattr(proxy, "_messages_client", _must_not_create_client)
    logged = []
    monkeypatch.setattr(proxy._logger, "log_proxy_call", lambda **kwargs: logged.append(kwargs))

    response = client.post(path, json={"contents": []})

    assert response.status_code == 400
    assert response.json()["error"]["status"] == "INVALID_ARGUMENT"
    # 0.7.6 invariant: every exit logs, even a local pre-payment reject.
    assert logged and logged[0]["http_status"] == 400
    assert "settlement_status" not in logged[0], "local reject never reached the gateway"


@pytest.mark.parametrize(
    "path",
    [
        "/v1beta/models/../../../v1/chat/completions:generateContent",
        "/v1beta/models/google/../../v1/videos/generations:generateContent",
    ],
)
def test_dot_segment_paths_never_reach_the_wallet(monkeypatch, client, path):
    """Literal ``..`` segments are normalized away by the HTTP layer before they
    reach the handler (404), so they never touch the wallet. Belt-and-suspenders
    alongside the in-handler slash rejection above."""
    monkeypatch.setattr(
        proxy,
        "_messages_client",
        lambda api_url: (_ for _ in ()).throw(AssertionError("wallet client must not be created")),
    )
    response = client.post(path, json={"contents": []})
    assert response.status_code in (400, 404)


def test_proxy_token_gate_denies_gemini_route(monkeypatch, client):
    """With BLOCKRUN_PROXY_TOKEN set, the route requires a Bearer token. The Google
    SDK's own credential (x-goog-api-key) must not satisfy the gate, and no wallet
    client is created for a denied request."""
    monkeypatch.setenv("BLOCKRUN_PROXY_TOKEN", "sekrit")
    monkeypatch.setattr(
        proxy,
        "_messages_client",
        lambda api_url: (_ for _ in ()).throw(AssertionError("must not reach wallet")),
    )

    denied = client.post(
        "/v1beta/models/gemini-2.5-flash:generateContent",
        json={"contents": []},
        headers={"x-goog-api-key": "sekrit"},
    )
    assert denied.status_code == 401

    wrong_bearer = client.post(
        "/v1beta/models/gemini-2.5-flash:generateContent",
        json={"contents": []},
        headers={"authorization": "Bearer nope"},
    )
    assert wrong_bearer.status_code == 401

