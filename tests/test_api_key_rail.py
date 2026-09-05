"""The API-key rail: a BlockRun account instead of an x402 wallet.

Two things are being pinned. First, that a key routes every surface at
``api.blockrun.ai`` with a Bearer header and no wallet anywhere in the path —
including the surfaces whose gateway spelling differs from the account API's.
Second, that the *absence* of on-chain cost is reported as absence: the wallet
rail's headline feature is the exact settled charge, and an estimate wearing
that label would be worse than no number at all.

Nothing here touches the network. The pooled httpx clients are replaced with
``httpx.MockTransport``, which is also what proves the requests carry the
headers and URLs claimed above.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import blockrun_litellm.proxy as proxy
from blockrun_litellm import _adapter, _apikey
from blockrun_llm.types import APIError, PaymentError

KEY = "brk_live_TESTKEY0123456789"

CHAT_BODY = {
    "id": "chatcmpl-acct-1",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "gpt-5.5-2026-04-23",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    "system_fingerprint": "fp_acct",
}


class Recorder:
    """A MockTransport that records every request it answers."""

    def __init__(self, responder):
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "no request was made"
        return self.requests[-1]


@pytest.fixture
def transport(monkeypatch):
    """Route both pooled clients through one recorder.

    Returns a factory: call it with a responder to install the behaviour for
    the test, and read ``.requests`` off the result.
    """

    def install(responder):
        rec = Recorder(responder)
        mock = httpx.MockTransport(rec.handle)
        sync = httpx.Client(transport=mock)
        aio = httpx.AsyncClient(transport=mock)
        monkeypatch.setattr(_apikey, "sync_http", lambda: sync)
        monkeypatch.setattr(_apikey, "async_http", lambda: aio)
        return rec

    return install


def json_responder(payload, status=200, headers=None):
    def _respond(_request):
        return httpx.Response(status, json=payload, headers=headers or {})

    return _respond


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


class TestResolution:
    def test_env_key_is_picked_up(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        assert _apikey.resolve_api_key() == KEY

    def test_explicit_key_beats_env(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_API_KEY", "brk_live_from_env")
        assert _apikey.resolve_api_key(KEY) == KEY

    @pytest.mark.parametrize(
        "wallet_key",
        [
            "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",  # Base hex
            "5JGiBQ8m1nP1cWQ1kR5nJTaHqZ1YvWc8Vc1sQ3sQ3sQ3sQ3sQ3sQ",  # Solana base58
            None,
            "",
        ],
    )
    def test_a_wallet_key_is_not_an_api_key(self, wallet_key):
        """The whole point of the prefix test: one argument, two credentials.

        LiteLLM's ``api_key`` lands in the adapter's ``private_key``. If a
        wallet key could be mistaken for an account key, every existing wallet
        caller would silently start authenticating against the account API with
        their private key in a header — which is the worst failure this package
        could have.
        """
        assert _apikey.looks_like_api_key(wallet_key) is False
        assert _apikey.resolve_api_key(wallet_key) is None

    def test_wallet_key_does_not_shadow_the_env_key(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        assert _apikey.resolve_api_key("0xdeadbeef") == KEY

    def test_no_credentials_means_the_wallet_rail(self):
        assert _apikey.resolve_api_key() is None

    def test_base_url_is_overridable(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_API_BASE_URL", "https://staging.example/")
        assert _apikey.api_base() == "https://staging.example"
        assert _apikey.target_url("/v1/models") == "https://staging.example/v1/models"

    def test_images_edits_is_rewritten_to_the_published_spelling(self):
        assert _apikey.target_url("/v1/images/edits").endswith("/v1/images/image2image")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_sync_chat_goes_to_the_account_api_with_a_bearer(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder(CHAT_BODY))

        payload = _adapter.chat_completion_sync(
            "openai/gpt-5.5", [{"role": "user", "content": "hi"}], max_tokens=8
        )

        assert str(rec.last.url) == "https://api.blockrun.ai/v1/chat/completions"
        assert rec.last.headers["authorization"] == f"Bearer {KEY}"
        sent = json.loads(rec.last.content)
        assert sent["model"] == "openai/gpt-5.5"
        assert sent["max_tokens"] == 8
        assert payload["choices"][0]["message"]["content"] == "hi"
        # Native passthrough survives the rail change.
        assert payload["system_fingerprint"] == "fp_acct"

    def test_no_wallet_is_ever_constructed(self, monkeypatch, transport):
        """A key must not reach any SDK client — those need a private key.

        Poisoning the constructors is the only assertion that actually proves
        it; a passing chat call would otherwise look identical if the wallet
        path happened to be reachable on the developer's machine.
        """
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(json_responder(CHAT_BODY))

        def _boom(*_a, **_kw):
            raise AssertionError("the API-key rail must not build a wallet client")

        monkeypatch.setattr(_adapter, "get_sync_client", _boom)
        monkeypatch.setattr(_adapter, "get_async_client", _boom)
        monkeypatch.setattr(_adapter, "get_image_client", _boom)

        _adapter.chat_completion_sync("openai/gpt-5.5", [{"role": "user", "content": "hi"}])

    def test_cost_is_absent_not_estimated(self, monkeypatch, transport):
        """No on-chain charge exists on this rail, and none is invented.

        The rail marker rides alongside so the audit row can say *why* the cost
        is missing — "billed to account credit", not "we failed to read it".
        """
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(json_responder(CHAT_BODY))

        payload = _adapter.chat_completion_sync(
            "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
        )
        meta = payload[_adapter._BLOCKRUN_META_KEY]
        assert meta == {"cost_usd": None, "settlement": None, "rail": "api_key"}

    def test_per_call_key_switches_rails_without_env(self, monkeypatch, transport):
        """``litellm.completion(..., api_key="brk_live_...")`` must just work."""
        rec = transport(json_responder(CHAT_BODY))
        _adapter.chat_completion_sync(
            "openai/gpt-5.5", [{"role": "user", "content": "hi"}], private_key=KEY
        )
        assert rec.last.headers["authorization"] == f"Bearer {KEY}"

    def test_async_chat(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder(CHAT_BODY))
        payload = asyncio.run(
            _adapter.chat_completion_async("openai/gpt-5.5", [{"role": "user", "content": "hi"}])
        )
        assert payload["id"] == "chatcmpl-acct-1"
        assert rec.last.headers["authorization"] == f"Bearer {KEY}"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


SSE = (
    b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}],'
    b'"system_fingerprint":"fp_acct"}\n\n'
    b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


class TestStreaming:
    def test_sse_becomes_sdk_chunks(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(
            lambda _r: httpx.Response(
                200, content=SSE, headers={"content-type": "text/event-stream"}
            )
        )
        chunks = list(
            _adapter.chat_completion_stream_sync(
                "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
            )
        )
        assert [c.choices[0].delta.content for c in chunks] == ["He", "llo"]
        # extra="allow" keeps the relay-detection signals the gateway forwards.
        assert chunks[0].model_extra["system_fingerprint"] == "fp_acct"

    def test_stream_asks_for_streaming(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(
            lambda _r: httpx.Response(
                200, content=SSE, headers={"content-type": "text/event-stream"}
            )
        )
        list(
            _adapter.chat_completion_stream_sync(
                "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
            )
        )
        assert json.loads(rec.last.content)["stream"] is True

    def test_a_bad_frame_does_not_kill_the_stream(self, monkeypatch, transport):
        """The caller has already paid for the completion; drop the frame, not it."""
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        broken = b"data: {not json}\n\n" + SSE
        transport(
            lambda _r: httpx.Response(
                200, content=broken, headers={"content-type": "text/event-stream"}
            )
        )
        chunks = list(
            _adapter.chat_completion_stream_sync(
                "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
            )
        )
        assert [c.choices[0].delta.content for c in chunks] == ["He", "llo"]

    def test_an_error_status_raises_instead_of_yielding_nothing(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(
            json_responder(
                {"error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"}},
                status=429,
            )
        )
        with pytest.raises(APIError) as excinfo:
            list(
                _adapter.chat_completion_stream_sync(
                    "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
                )
            )
        assert excinfo.value.status_code == 429


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_402_is_a_payment_error_pointing_at_billing(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(
            json_responder(
                {
                    "error": {
                        "message": "Balance exhausted — add credit to continue",
                        "type": "insufficient_quota",
                        "code": "quota_exceeded",
                    }
                },
                status=402,
            )
        )
        with pytest.raises(PaymentError) as excinfo:
            _adapter.chat_completion_sync("openai/gpt-5.5", [{"role": "user", "content": "hi"}])
        assert "user.blockrun.ai" in str(excinfo.value)

    def test_401_names_the_portal(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(
            json_responder(
                {"error": {"message": "Invalid API key", "code": "invalid_api_key"}}, status=401
            )
        )
        with pytest.raises(APIError) as excinfo:
            _adapter.chat_completion_sync("openai/gpt-5.5", [{"role": "user", "content": "hi"}])
        assert "user.blockrun.ai" in str(excinfo.value)
        assert excinfo.value.status_code == 401

    def test_html_from_a_load_balancer_is_not_echoed_as_a_message(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(lambda _r: httpx.Response(502, content=b"<html>Bad Gateway</html>"))
        with pytest.raises(APIError) as excinfo:
            _adapter.chat_completion_sync("openai/gpt-5.5", [{"role": "user", "content": "hi"}])
        assert "<html>" not in str(excinfo.value)
        assert "502" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


IMAGE_BODY = {"created": 1, "data": [{"url": "https://cdn.example/a.png"}]}


class TestMedia:
    def test_image_generation_sends_the_sdk_body(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder(IMAGE_BODY))
        result = _adapter.image_generation_sync("a cat")
        assert result == IMAGE_BODY
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/images/generations"
        sent = json.loads(rec.last.content)
        # Defaults come off the SDK class, so the two rails send the same body
        # for a caller who named neither model nor size.
        assert sent == {"model": "google/nano-banana", "prompt": "a cat", "size": "1024x1024", "n": 1}

    def test_image_quality_is_forwarded_because_there_is_no_chain(self, monkeypatch, transport):
        """The Solana-only rule is a property of the gateways, not of accounts."""
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder(IMAGE_BODY))
        _adapter.image_generation_sync("a cat", model="openai/gpt-image-2", quality="high")
        assert json.loads(rec.last.content)["quality"] == "high"

    def test_image_edit_uses_the_published_path(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder(IMAGE_BODY))
        _adapter.image_edit_sync("make it blue", "https://cdn.example/in.png")
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/images/image2image"
        assert json.loads(rec.last.content)["image"] == "https://cdn.example/in.png"

    def test_speech(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder({"data": [{"url": "https://cdn.example/a.mp3"}]}))
        asyncio.run(_adapter.speech_generation_async("hello", voice="george"))
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/audio/speech"
        assert json.loads(rec.last.content)["voice"] == "george"

    def test_music_rejects_lyrics_with_instrumental_before_the_wire(self, monkeypatch, transport):
        """The SDK raises this locally; the account rail must not lose the check
        and let the gateway bill for a request it will refuse."""
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        rec = transport(json_responder({}))
        with pytest.raises(ValueError):
            asyncio.run(
                _adapter.music_generation_async("a song", instrumental=True, lyrics="la la")
            )
        assert rec.requests == []

    def test_video_submits_then_polls_to_completion(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        done = {"id": "job-1", "status": "completed", "data": [{"url": "https://cdn/x.mp4"}]}
        states = [
            httpx.Response(200, json={"id": "job-1", "status": "queued"}),
            httpx.Response(202, json={"id": "job-1", "status": "in_progress"}),
            httpx.Response(200, json=done),
        ]
        transport(lambda _r: states.pop(0))
        monkeypatch.setattr(_apikey, "_POLL_INTERVAL_S", 0)

        result = asyncio.run(_adapter.video_generation_async("a cat", model="xai/grok-imagine-video"))
        assert result == done

    def test_video_failure_is_raised_not_returned(self, monkeypatch, transport):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        states = [
            httpx.Response(200, json={"id": "job-1", "status": "queued"}),
            httpx.Response(200, json={"id": "job-1", "status": "failed", "error": "nsfw"}),
        ]
        transport(lambda _r: states.pop(0))
        monkeypatch.setattr(_apikey, "_POLL_INTERVAL_S", 0)

        with pytest.raises(APIError) as excinfo:
            asyncio.run(_adapter.video_generation_async("a cat", model="xai/grok-imagine-video"))
        assert "nsfw" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Proxy routes
# ---------------------------------------------------------------------------


@pytest.fixture
def proxy_client(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    return TestClient(proxy.app)


class TestProxy:
    def test_chat_passthrough_swaps_in_the_key(self, proxy_client, transport):
        rec = transport(json_responder(CHAT_BODY))
        response = proxy_client.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer a-local-proxy-secret"},
        )
        assert response.status_code == 200
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/chat/completions"
        assert rec.last.headers["authorization"] == f"Bearer {KEY}"

    def test_the_clients_own_secret_never_leaves_the_host(self, proxy_client, transport):
        """BLOCKRUN_PROXY_TOKEN guards the sidecar locally. Forwarding it to a
        remote host would leak the credential it exists to protect."""
        rec = transport(json_responder(CHAT_BODY))
        proxy_client.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer a-local-proxy-secret"},
        )
        assert "a-local-proxy-secret" not in str(dict(rec.last.headers))

    def test_anthropic_messages_passthrough(self, proxy_client, transport):
        rec = transport(json_responder({"id": "msg_1", "type": "message", "role": "assistant"}))
        response = proxy_client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4.5",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 200
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/messages"

    def test_models_comes_from_the_account_catalogue(self, proxy_client, transport):
        catalogue = {"object": "list", "data": [{"id": "openai/gpt-5.5", "object": "model"}]}
        rec = transport(json_responder(catalogue))
        response = proxy_client.get("/v1/models")
        assert response.status_code == 200
        assert response.json() == catalogue
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/models"

    def test_native_gemini_is_a_501_that_says_what_to_do(self, proxy_client, transport):
        rec = transport(json_responder({}))
        response = proxy_client.post(
            "/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert response.status_code == 501
        body = response.json()["error"]
        assert body["code"] == "unsupported_on_api_key"
        assert "wallet" in body["message"]
        assert rec.requests == [], "a 501 must not cost a round trip"

    def test_image_route_reaches_the_published_path(self, proxy_client, transport):
        rec = transport(json_responder(IMAGE_BODY))
        response = proxy_client.post(
            "/v1/images/edits",
            json={
                "prompt": "make it blue",
                "image": "https://cdn.example/in.png",
                "model": "openai/gpt-image-2",
            },
        )
        assert response.status_code == 200
        assert str(rec.last.url) == "https://api.blockrun.ai/v1/images/image2image"

    def test_a_402_reaches_the_client_as_a_402(self, proxy_client, transport):
        transport(
            json_responder(
                {"error": {"message": "Balance exhausted", "code": "quota_exceeded"}}, status=402
            )
        )
        response = proxy_client.post(
            "/v1/images/generations", json={"prompt": "a cat", "model": "google/nano-banana"}
        )
        assert response.status_code == 402
        assert "user.blockrun.ai" in json.dumps(response.json())


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    """`cost_source` has to distinguish "no charge exists" from "we missed it"."""

    def test_the_provider_marks_the_rail_on_the_response(self, monkeypatch, transport):
        """The marker the audit row reads, put there by the non-streaming path.

        Asserted on ``_hidden_params`` rather than through ``litellm.callbacks``
        on purpose: LiteLLM caches its callback list internally, so a
        callback-driven assertion here passes alone and fails after any other
        test has touched that list. The two halves — marker written, marker
        read — are pinned separately instead.
        """
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        transport(json_responder(CHAT_BODY))

        from blockrun_litellm.provider import BlockRunLLM

        response = BlockRunLLM().completion(
            "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
        )
        assert response._hidden_params["blockrun_rail"] == "api_key"
        # No charge was invented to carry it.
        assert "blockrun_cost_usd" not in response._hidden_params

        from blockrun_litellm import logger as _logger

        real = _logger._extract_real_cost(response)
        assert real == {"cost_usd": None, "settlement": None, "rail": "api_key"}
        assert _logger._cost_fields(real, 0.0009)["cost_source"] == "blockrun_account"

    def test_proxy_rows_are_tagged_too(self, monkeypatch, tmp_path, transport):
        """The sidecar has no LiteLLM callback to ride; it logs for itself."""
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        log = tmp_path / "proxy.jsonl"
        monkeypatch.setenv("BLOCKRUN_LITELLM_LOG", str(log))
        transport(json_responder(CHAT_BODY))

        client = TestClient(proxy.app)
        client.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
        )

        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        assert rows[-1]["cost_source"] == "blockrun_account"
        assert rows[-1]["cost_usd"] is None

    def test_wallet_calls_keep_their_own_tags(self, tmp_path):
        """The new value must not leak onto the rail that reports real charges."""
        from blockrun_litellm import logger as _logger

        real = {"cost_usd": 0.0021, "settlement": {"tx_hash": "0xabc"}, "rail": None}
        assert _logger._cost_fields(real, 0.0009)["cost_source"] == "blockrun_x402"
        assert _logger._cost_fields(real, 0.0009)["cost_usd"] == 0.0021

        none_reported = {"cost_usd": None, "settlement": None, "rail": None}
        assert _logger._cost_fields(none_reported, 0.0009) == {
            "cost_usd": 0.0009,
            "cost_source": "litellm_estimate",
        }

    def test_account_row_carries_the_estimate_not_a_settled_number(self):
        from blockrun_litellm import logger as _logger

        fields = _logger._cost_fields(
            {"cost_usd": None, "settlement": None, "rail": "api_key"}, 0.0009
        )
        assert fields == {"cost_usd": 0.0009, "cost_source": "blockrun_account"}
