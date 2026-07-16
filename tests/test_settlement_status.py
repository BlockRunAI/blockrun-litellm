"""A failed call must not be logged as free when it might have been charged.

`cost_usd=None` reads as "$0" but is also what we write when we don't know.
Those are different facts, and on Solana the difference is real money: several
routes (chat, search, both image routes, music) settle **optimistically** —
settle fires in parallel with the upstream work — so a call that verifies and
then fails upstream IS charged. Its error response carries no settlement header,
so "no proof of payment" and "you were charged" co-occur exactly when it matters.

**Every** Solana failure counts, not just 5xx. The settle fires first, so the
charged failures come back as 4xx: content filter → 400, rate limit → 429, and a
poll → 402 after the POST already settled. An earlier cut gated on `>= 500` and
missed all of them; the tests below exist to keep that from coming back.

Base *media* routes settle only after a successful upstream call, so a failure
there is genuinely free and must NOT be flagged — noise gets ignored, and an
ignored flag costs us the Solana signal too. Two Base exceptions: a 504 is our
own await ceiling (the worker keeps running and can still settle), and a body
that won't parse means the gateway already settled.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from blockrun_llm.types import APIError, PaymentError

import blockrun_litellm.proxy as proxy

# Base settles before responding, so a success carries the tx hash.
OK_RESULT_BASE: Dict[str, Any] = {
    "created": 1,
    "model": "openai/gpt-image-2",
    "data": [{"url": "https://cdn/x.png"}],
    "txHash": "0xabc",
}

# Solana settles OPTIMISTICALLY, so at response time there is no tx hash to
# report — the gateway sends `payment: {status: "settling"}` with the comment
# "Settlement is optimistic, so there is no tx hash yet". A fixture that carries
# a 0x hash here is a Base response wearing a Solana label, and it manufactures
# confidence that Solana successes come with proof of payment. They do not.
OK_RESULT_SOLANA: Dict[str, Any] = {
    "created": 1,
    "model": "openai/gpt-image-2",
    "data": [{"url": "https://cdn/x.png"}],
    "payment": {"status": "settling", "network": "solana"},
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
    """EVERY Solana failure is flagged, not just 5xx.

    An earlier cut gated on `>= 500`, reasoning that a 4xx is the gateway
    refusing before settle. That is false, and it was the expensive kind of
    wrong: the optimistic settle fires FIRST, so the charged failures come back
    as 4xx — content filter → 400 (blockrun-sol images/generations:376), rate
    limit → 429 (:359). Both routine, both client-triggerable, both charged.
    """

    @pytest.mark.parametrize(
        "label,exc",
        [
            ("upstream 5xx", APIError("upstream boom", 500)),
            # The one the >=500 gate missed. A prompt that trips the content
            # filter is the single most common charged failure on this chain.
            ("content policy → 400", APIError("Content policy violation", 400)),
            ("rate limit → 429", APIError("Rate limit exceeded", 429)),
            ("provider bad request → 400", APIError("bad request", 400)),
            # Solana settles at POST; the poll route returns 402 on wallet
            # binding AFTER the money moved ("the POST already settled").
            (
                "poll 402 after settle",
                PaymentError("rejected", status_code=402, response={"details": "binding"}),
            ),
        ],
    )
    def test_every_failure_is_flagged_unknown(self, client, monkeypatch, log_path, label, exc):
        _solana(monkeypatch)
        _call(client, monkeypatch, side_effect=exc)
        row = _last_row(log_path)
        assert row["settlement_status"] == "unknown", (
            f"{label}: optimistic settle means this may have been charged — "
            "cost_usd=None with no flag reads as $0 and hides a real debit"
        )
        assert row["cost_usd"] is None  # unknown, just no longer implied to be zero


class TestBaseFailuresAreFree:
    @pytest.mark.parametrize(
        "label,exc",
        [
            ("upstream 5xx", APIError("upstream boom", 500)),
            ("gateway 400", APIError("bad model", 400)),
            ("payment rejected", PaymentError("rejected", status_code=402, response={})),
        ],
    )
    def test_failures_are_not_flagged(self, client, monkeypatch, log_path, label, exc):
        """Base media routes settle only after a successful upstream call, so a
        failure is genuinely free. Flagging every Base error would be noise, and
        noise gets ignored — which would cost us the Solana signal too.
        """
        _base(monkeypatch)
        _call(client, monkeypatch, side_effect=exc)
        assert "settlement_status" not in _last_row(log_path), f"{label} is free on Base"

    def test_our_own_await_ceiling_is_flagged(self, client, monkeypatch, log_path):
        """504 is _run_media giving up on the wait — it abandons the await but
        leaves the worker running, so the call can still complete and settle.
        The only Base failure we cannot call free.
        """
        _base(monkeypatch)
        _call(client, monkeypatch, side_effect=APIError("await ceiling exceeded", 504))
        assert _last_row(log_path)["settlement_status"] == "unknown"


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
    def test_base_success_carries_the_tx_and_no_flag(self, client, monkeypatch, log_path):
        _base(monkeypatch)
        response = _call(client, monkeypatch, return_value=dict(OK_RESULT_BASE))
        assert response.status_code == 200
        row = _last_row(log_path)
        assert "settlement_status" not in row, "proven settled needs no flag — the tx is right there"
        assert row["settlement"]["tx_hash"] == "0xabc"

    def test_solana_success_has_no_tx_and_no_flag(self, client, monkeypatch, log_path):
        """A real Solana success carries NO tx hash — settle is still in flight.

        So the row is `cost_usd=None, settlement=None`, which looks exactly like
        a free call. This test exists to state that plainly rather than paper
        over it with a Base-shaped fixture: `settlement_status` is about
        *failures*, and it does not fix the fact that successful Solana spend is
        also absent from this ledger. The gateway's own ledger is the authority.
        """
        _solana(monkeypatch)
        response = _call(client, monkeypatch, return_value=dict(OK_RESULT_SOLANA))
        assert response.status_code == 200
        row = _last_row(log_path)
        assert row["settlement"] is None, "optimistic settle: no tx hash exists yet"
        assert "settlement_status" not in row, "a success is not the failure case this flags"


class TestEveryExitLogs:
    """A media call that moves money and leaves NO audit row is invisible to
    reconciliation — strictly worse than one recorded with an unknown cost.

    Both holes below produced exactly that: the arm returned via `raise`, or no
    arm matched at all, so log_proxy_call never ran.
    """

    def test_malformed_paid_body_is_logged_not_blamed_on_the_caller(
        self, client, monkeypatch, log_path
    ):
        """json.JSONDecodeError IS a ValueError.

        The SDK parses the paid 200 with a bare `.json()`, so a truncated body
        raised JSONDecodeError → the ValueError arm → HTTPException(400) →
        `raise` → no row. The gateway had already settled: money moved, the
        caller got blamed for it, and reconciliation never saw the call.
        Catching ValidationError first did not help — `.json()` fails before
        pydantic is ever reached.
        """
        _base(monkeypatch)
        exc = json.JSONDecodeError("Expecting value", "<truncated body>", 0)
        assert isinstance(exc, ValueError), "premise of this test"
        response = _call(client, monkeypatch, side_effect=exc)

        assert response.status_code == 502, "upstream's fault, not the caller's"
        row = _last_row(log_path)
        assert row["settlement_status"] == "unknown", "settle ran before the parse failed"

    def test_transport_error_after_payment_still_logs(self, client, monkeypatch, log_path):
        """httpx.ReadTimeout escapes the SDK unwrapped and matched no arm, so a
        timeout on a 10-minute image call whose optimistic settle had already
        fired left no row at all. A timeout is the likeliest failure on that route.
        """
        _solana(monkeypatch)
        response = _call(client, monkeypatch, side_effect=httpx.ReadTimeout("timed out"))
        assert response.status_code == 502
        assert _last_row(log_path)["settlement_status"] == "unknown"

    def test_local_validation_error_still_logs_but_is_not_flagged(
        self, client, monkeypatch, log_path
    ):
        """SDK request validation is local and pre-payment: 400, a row, no flag.

        It must still produce a row (the invariant is every exit logs) and must
        keep the {"detail": ...} shape callers already parse.
        """
        _solana(monkeypatch)
        response = _call(
            client, monkeypatch, side_effect=ValueError("Cannot specify lyrics when instrumental")
        )
        assert response.status_code == 400
        assert "lyrics" in response.json()["detail"], "response shape must not change"
        row = _last_row(log_path)
        assert row["http_status"] == 400
        assert "settlement_status" not in row, "never reached the gateway — genuinely free"


class TestChainIsSnapshotted:
    def test_env_flip_mid_call_cannot_reclassify_a_solana_charge_as_free(
        self, client, monkeypatch, log_path
    ):
        """BLOCKRUN_API_URL is a mutable global and media calls run for minutes.

        Resolving the chain at LOG time meant a flip in flight would classify a
        Solana charge as Base and write it off as $0 — the exact false negative
        this flag exists to prevent, reintroduced by a race.
        """
        _solana(monkeypatch)

        async def fail_then_flip(**_kwargs):
            # The call started on Solana; the env flips to Base before we log.
            os.environ.pop("BLOCKRUN_API_URL", None)
            raise APIError("upstream boom", 500)

        monkeypatch.setattr(proxy._adapter, "image_generation_async", fail_then_flip)
        client.post("/v1/images/generations", json={"prompt": "a cat"})

        assert _last_row(log_path)["settlement_status"] == "unknown", (
            "the call ran on Solana and may have been charged — a late env read "
            "must not turn it into a free Base failure"
        )
