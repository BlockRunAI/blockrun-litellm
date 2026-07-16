"""A failed call must not be logged as free when it might have been charged.

`cost_usd=None` reads as "$0" but is also what we write when we don't know.
Those are different facts, and on Solana the difference is real money: several
routes (chat, search, both image routes, music) settle **optimistically** —
settle fires in parallel with the upstream work — so a call that verifies and
then fails upstream IS charged. Its error response carries no settlement header,
so "no proof of payment" and "you were charged" co-occur exactly when it matters.

Base settles only after a successful upstream call, so a failure there is
genuinely free and must NOT be flagged — a flag on every Base error would be
noise, and noise gets ignored.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from blockrun_llm.types import APIError, PaymentError

import blockrun_litellm.proxy as proxy

OK_RESULT: Dict[str, Any] = {
    "created": 1,
    "model": "openai/gpt-image-2",
    "data": [{"url": "https://cdn/x.png"}],
    "txHash": "0xabc",
}


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("BLOCKRUN_LITELLM_LOG", str(path))
    return path


@pytest.fixture
def client():
    return TestClient(proxy.app)


def _solana(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_URL", "https://sol.blockrun.ai/api")


def _base(monkeypatch):
    monkeypatch.delenv("BLOCKRUN_API_URL", raising=False)


def _last_row(path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows, "expected an audit row"
    return rows[-1]


def _call(client, monkeypatch, **mock_kwargs):
    monkeypatch.setattr(proxy._adapter, "image_generation_async", AsyncMock(**mock_kwargs))
    return client.post("/v1/images/generations", json={"prompt": "a cat"})


def _a_validation_error() -> ValidationError:
    class M(BaseModel):
        x: int

    try:
        M(x="not an int")
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


class TestSolanaChargedFailures:
    def test_upstream_failure_is_flagged_unknown(self, client, monkeypatch, log_path):
        """The case that loses money silently: optimistic settle + 5xx."""
        _solana(monkeypatch)
        response = _call(client, monkeypatch, side_effect=APIError("upstream boom", 500))
        assert response.status_code == 500
        row = _last_row(log_path)
        assert row["settlement_status"] == "unknown", (
            "a Solana 5xx can still have settled — logging cost_usd=None with no "
            "flag reads as $0 and hides a real charge"
        )
        assert row["cost_usd"] is None  # still unknown, just no longer implied to be zero

    def test_gateway_refusal_is_not_flagged(self, client, monkeypatch, log_path):
        """4xx = refused before settle, on either chain. Flagging it is noise."""
        _solana(monkeypatch)
        _call(client, monkeypatch, side_effect=APIError("bad model", 400))
        assert "settlement_status" not in _last_row(log_path)

    def test_payment_rejected_is_not_flagged(self, client, monkeypatch, log_path):
        """PaymentError means the payment itself never landed."""
        _solana(monkeypatch)
        _call(
            client,
            monkeypatch,
            side_effect=PaymentError("rejected", status_code=402, response={"details": "no funds"}),
        )
        assert "settlement_status" not in _last_row(log_path)


class TestBaseFailuresAreFree:
    def test_upstream_failure_is_not_flagged(self, client, monkeypatch, log_path):
        """Base settles only after a successful upstream call — a 5xx costs nothing."""
        _base(monkeypatch)
        _call(client, monkeypatch, side_effect=APIError("upstream boom", 500))
        assert "settlement_status" not in _last_row(log_path)


class TestChargedOnBothChains:
    @pytest.mark.parametrize("chain", ["base", "solana"])
    def test_parse_failure_after_settlement_is_always_flagged(
        self, client, monkeypatch, log_path, chain
    ):
        """The gateway settled and returned 200; only the SDK's parse failed.

        The money moved regardless of chain, so the chain check must not gate
        this one.
        """
        (_solana if chain == "solana" else _base)(monkeypatch)
        response = _call(client, monkeypatch, side_effect=_a_validation_error())
        assert response.status_code == 502
        assert _last_row(log_path)["settlement_status"] == "unknown"


class TestSuccessRowsUnchanged:
    @pytest.mark.parametrize("chain", ["base", "solana"])
    def test_success_carries_the_tx_and_no_flag(self, client, monkeypatch, log_path, chain):
        (_solana if chain == "solana" else _base)(monkeypatch)
        response = _call(client, monkeypatch, return_value=dict(OK_RESULT))
        assert response.status_code == 200
        row = _last_row(log_path)
        assert "settlement_status" not in row, "proven settled needs no flag — the tx is right there"
        assert row["settlement"]["tx_hash"] == "0xabc"
