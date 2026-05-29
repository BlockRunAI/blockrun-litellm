"""Tests for the proxy's PaymentError surfacing.

Regression guard: when blockrun-llm raises a PaymentError carrying the
gateway's original failure JSON (e.g. ``{"error": "Payment settlement
failed", "details": "transaction_simulation_failed"}``), the sidecar
must pass the ``details`` through to the HTTP 402 body so customers
see the real reason instead of just our generic SDK message.
"""

from __future__ import annotations

from blockrun_llm.types import PaymentError
from blockrun_litellm.proxy import (
    _payment_error_payload,
    _payment_error_sse_message,
)


class TestPaymentErrorPayload:
    def test_preserves_gateway_details(self) -> None:
        """The customer-facing payload should include the gateway's
        ``details`` field verbatim — that's the whole point of the fix."""
        exc = PaymentError(
            "Payment rejected by gateway: transaction_simulation_failed",
            status_code=402,
            response={
                "message": "Payment settlement failed",
                "details": "transaction_simulation_failed",
            },
        )
        payload = _payment_error_payload(exc)
        assert payload["error"] == (
            "Payment rejected by gateway: transaction_simulation_failed"
        )
        assert payload["details"] == "transaction_simulation_failed"

    def test_omits_details_when_absent(self) -> None:
        """Pre-0.32.0 PaymentError without response — keep the legacy shape
        so we don't break clients that parsed the old ``{"error": "..."}``."""
        exc = PaymentError("Payment rejected")
        payload = _payment_error_payload(exc)
        assert payload == {"error": "Payment rejected"}

    def test_omits_non_string_details(self) -> None:
        """If the SDK somehow surfaces a non-string details (defensive),
        don't pass through garbage — fall back to message-only."""
        exc = PaymentError(
            "Payment rejected",
            status_code=402,
            response={"message": "Payment settlement failed", "details": ["x"]},
        )
        payload = _payment_error_payload(exc)
        assert "details" not in payload


class TestPaymentErrorSseMessage:
    def test_folds_details_into_message(self) -> None:
        """SSE streams only get one error message field per event — fold
        the gateway reason into it so streaming clients see it too."""
        exc = PaymentError(
            "Payment rejected",
            status_code=402,
            response={"details": "transaction_simulation_failed"},
        )
        msg = _payment_error_sse_message(exc)
        assert "transaction_simulation_failed" in msg

    def test_avoids_duplication_when_already_in_message(self) -> None:
        """If the message already mentions the reason (from the SDK helper),
        don't append it twice."""
        exc = PaymentError(
            "Payment rejected by gateway: transaction_simulation_failed",
            status_code=402,
            response={"details": "transaction_simulation_failed"},
        )
        msg = _payment_error_sse_message(exc)
        assert msg.count("transaction_simulation_failed") == 1

    def test_no_response_no_change(self) -> None:
        """Backwards-compatible path: no response → return str(exc) as-is."""
        exc = PaymentError("Payment rejected")
        assert _payment_error_sse_message(exc) == "Payment rejected"
