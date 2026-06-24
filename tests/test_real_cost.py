"""The real x402 charge must surface through the provider and the logger,
overriding LiteLLM's token×list-price estimate.

Regression guard for the customer-reported gap: LiteLLM reported ~$0.00x while
the wallet was debited ~$0.0x. The SDK now attaches the real per-call charge to
the response (``ChatResponse.cost_usd``); these tests lock in that we (a) route
it into ``_hidden_params['response_cost']`` so LiteLLM bills the real amount,
(b) expose it as ``blockrun_cost_usd`` + ``blockrun_settlement``, and (c) record
it in the JSONL audit row as ``cost_usd`` with ``cost_source='blockrun_x402'``.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import litellm
import pytest

from blockrun_litellm import BlockRunLLM
from blockrun_litellm.logger import _build_entry

from .conftest import make_chat_response

REAL_COST = 0.06354  # what the wallet was actually debited
SETTLEMENT = {"tx_hash": "0xabc123", "amount_micro_usdc": "63540", "network": "eip155:8453"}


def _stub(monkeypatch, *, cost_usd=REAL_COST, settlement=SETTLEMENT):
    resp = make_chat_response()
    if cost_usd is not None:
        resp.cost_usd = cost_usd
    if settlement is not None:
        resp.settlement = settlement
    mock = MagicMock()
    mock.chat_completion.return_value = resp
    mock._last_call_cost = cost_usd
    monkeypatch.setattr("blockrun_litellm._adapter.get_sync_client", lambda **_: mock)
    return mock


def test_real_cost_overrides_litellm_estimate(monkeypatch) -> None:
    _stub(monkeypatch)
    resp = BlockRunLLM().completion(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 32},
    )
    hp = resp._hidden_params
    assert hp["response_cost"] == pytest.approx(REAL_COST)
    assert hp["blockrun_cost_usd"] == pytest.approx(REAL_COST)
    assert hp["blockrun_settlement"] == SETTLEMENT


def test_logger_records_real_cost_and_source(monkeypatch) -> None:
    _stub(monkeypatch)
    resp = BlockRunLLM().completion(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 32},
    )
    kwargs = {"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}
    entry = _build_entry(kwargs, resp, time.time(), time.time())
    assert entry["cost_usd"] == pytest.approx(REAL_COST)
    assert entry["cost_source"] == "blockrun_x402"
    assert entry["settlement"] == SETTLEMENT
    # token usage is still real and preserved
    assert entry["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_free_call_records_zero_not_stale(monkeypatch) -> None:
    # Free/cached call: SDK attaches cost_usd=0.0; must not inherit a prior charge.
    _stub(monkeypatch, cost_usd=0.0, settlement=None)
    resp = BlockRunLLM().completion(
        model="nvidia/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 32},
    )
    assert resp._hidden_params["blockrun_cost_usd"] == 0.0
    entry = _build_entry({"model": "blockrun/x"}, resp, time.time(), time.time())
    assert entry["cost_usd"] == 0.0
    assert entry["cost_source"] == "blockrun_x402"


def test_falls_back_to_estimate_when_no_real_cost(monkeypatch) -> None:
    # Older SDK that attaches nothing and exposes no _last_call_cost -> estimate path.
    _stub(monkeypatch, cost_usd=None, settlement=None)
    resp = BlockRunLLM().completion(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 32},
    )
    assert "blockrun_cost_usd" not in resp._hidden_params
    # Simulate LiteLLM having computed its own estimate.
    resp._hidden_params["response_cost"] = 0.0071
    entry = _build_entry({"model": "blockrun/openai/gpt-5.5"}, resp, time.time(), time.time())
    assert entry["cost_usd"] == pytest.approx(0.0071)
    assert entry["cost_source"] == "litellm_estimate"
    assert entry["estimated_cost_usd"] == pytest.approx(0.0071)
