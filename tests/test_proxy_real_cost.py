"""Real x402 cost on the FastAPI sidecar passthrough (#12, proxy half).

The raw ``/v1/chat/completions`` + ``/v1/messages`` passthrough relays bytes and
never went through ``_build_response``, so it surfaced no cost. The gateway
returns the real on-chain charge in the ``X-PAYMENT-RESPONSE`` response header on
the paid call (per-response → race-free, no shared-transport correlation). We:

  * decode it (``decode_settlement_header`` → amount_micro_usdc → USD), and
  * surface it as ``x-blockrun-cost-usd`` / ``x-blockrun-settlement`` response
    headers (always), plus an opt-in JSONL audit row (BLOCKRUN_LITELLM_LOG).

These tests stub the cached signing client with a canned upstream response and
assert the cost is surfaced on both the streaming and non-streaming paths.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import blockrun_litellm.proxy as proxy

REAL_COST = 0.06354
AMOUNT_MICRO = 63540  # 0.06354 USDC in micro-USDC


def _payment_response_header(amount_micro: int = AMOUNT_MICRO) -> str:
    payload = {
        "amount": amount_micro,
        "transaction": "0xabc123",
        "network": "eip155:8453",
        "payer": "0xpayer",
        "payee": "0xpayee",
        "success": True,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


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

    def build_request(self, method, url, content=None, headers=None):
        return httpx.Request(method, url, content=content, headers=headers)

    def send(self, request, stream=False):
        return self._response

    def post(self, url, content=None, headers=None):
        return self._response


@pytest.fixture
def client() -> TestClient:
    return TestClient(proxy.app)


def _patch(monkeypatch, response: httpx.Response) -> None:
    monkeypatch.setattr(proxy, "_messages_client", lambda api_url: _FakeClient(response))


# ---------------------------------------------------------------------------
# Response headers — non-stream + stream
# ---------------------------------------------------------------------------

def test_non_stream_surfaces_cost_header(monkeypatch, client):
    ok = httpx.Response(
        200,
        headers={
            "content-type": "application/json",
            "x-payment-response": _payment_response_header(),
        },
        content=b'{"id":"chatcmpl_1","object":"chat.completion"}',
    )
    _patch(monkeypatch, ok)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.content == b'{"id":"chatcmpl_1","object":"chat.completion"}'  # body untouched
    assert float(resp.headers["x-blockrun-cost-usd"]) == pytest.approx(REAL_COST)
    settlement = json.loads(resp.headers["x-blockrun-settlement"])
    assert settlement["tx_hash"] == "0xabc123"
    assert settlement["network"] == "eip155:8453"


def test_streaming_surfaces_cost_header(monkeypatch, client):
    ok = httpx.Response(
        200,
        headers={
            "content-type": "text/event-stream",
            "x-payment-response": _payment_response_header(),
        },
        stream=_BytesStream(b"event: message_start\ndata: {}\n\n"),
    )
    _patch(monkeypatch, ok)

    resp = client.post("/v1/messages", json={"model": "claude-haiku-4-5", "stream": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert b"message_start" in resp.content  # body still streamed
    assert float(resp.headers["x-blockrun-cost-usd"]) == pytest.approx(REAL_COST)


def test_no_payment_header_no_cost_header(monkeypatch, client):
    # Free/cached call: no X-PAYMENT-RESPONSE -> no cost header (graceful).
    ok = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=b'{"id":"chatcmpl_free"}',
    )
    _patch(monkeypatch, ok)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nvidia/deepseek-v4-flash", "messages": []},
    )
    assert resp.status_code == 200
    assert "x-blockrun-cost-usd" not in resp.headers


# ---------------------------------------------------------------------------
# Opt-in JSONL audit row (BLOCKRUN_LITELLM_LOG)
# ---------------------------------------------------------------------------

def test_audit_row_written_when_log_enabled(monkeypatch, tmp_path, client):
    log_path = tmp_path / "proxy_calls.jsonl"
    monkeypatch.setenv("BLOCKRUN_LITELLM_LOG", str(log_path))
    ok = httpx.Response(
        200,
        headers={
            "content-type": "application/json",
            "x-payment-response": _payment_response_header(),
        },
        content=b'{"id":"chatcmpl_1"}',
    )
    _patch(monkeypatch, ok)

    client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
    )

    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "openai/gpt-5.5"
    assert row["mode"] == "proxy_passthrough"
    assert row["cost_usd"] == pytest.approx(REAL_COST)
    assert row["cost_source"] == "blockrun_x402"
    assert row["stream"] is False
    assert row["settlement"]["tx_hash"] == "0xabc123"


def test_no_audit_when_log_disabled(monkeypatch, tmp_path, client):
    # Default: no BLOCKRUN_LITELLM_LOG -> the sidecar writes nothing to disk.
    monkeypatch.delenv("BLOCKRUN_LITELLM_LOG", raising=False)
    written = {"n": 0}
    monkeypatch.setattr(proxy._logger, "_write_entry", lambda *a, **k: written.__setitem__("n", written["n"] + 1))
    ok = httpx.Response(
        200,
        headers={
            "content-type": "application/json",
            "x-payment-response": _payment_response_header(),
        },
        content=b"{}",
    )
    _patch(monkeypatch, ok)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-5.5", "messages": []},
    )
    # Header is still surfaced; only the disk audit is gated off.
    assert float(resp.headers["x-blockrun-cost-usd"]) == pytest.approx(REAL_COST)
    assert written["n"] == 0
