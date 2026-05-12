"""Unit tests for the local JSONL logger.

We don't hit the live BlockRun gateway here — instead we drive the
:class:`JSONLLogger` callback directly with synthetic ``ModelResponse``
shapes that match what LiteLLM actually emits in each of the four
quadrants (sync/async × success/failure).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from blockrun_litellm.logger import (
    JSONLLogger,
    _build_entry,
    enable_local_logging,
    proxy_logger,
)


# ---------------------------------------------------------------------------
# Synthetic LiteLLM shapes
# ---------------------------------------------------------------------------


def _fake_response(
    *, content: str = "Hello world.", prompt_tokens: int = 10,
    completion_tokens: int = 3, cost: float | None = 0.00012,
):
    """Build a ModelResponse-shaped namespace with usage + content."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        _hidden_params={"response_cost": cost} if cost is not None else {},
    )


def _empty_stream_chunk():
    """Shape LiteLLM passes during streaming before the final fire — null
    usage, null content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=None,
        _hidden_params={},
    )


# ---------------------------------------------------------------------------
# _build_entry tests
# ---------------------------------------------------------------------------


class TestBuildEntry:
    def test_success_non_stream_captures_full_payload(self):
        entry = _build_entry(
            {"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            _fake_response(content="Pong.", prompt_tokens=5, completion_tokens=2),
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 1, 1, 0, 0, 1),
        )
        assert entry is not None
        assert entry["status"] == "success"
        assert entry["model"] == "blockrun/openai/gpt-5.5"
        assert entry["provider"] == "blockrun"
        assert entry["stream"] is False
        assert entry["completion"] == "Pong."
        assert entry["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        assert entry["cost_usd"] == pytest.approx(0.00012)
        assert entry["latency_ms"] == pytest.approx(1000.0)
        assert entry["messages"] == [{"role": "user", "content": "hi"}]

    def test_intermediate_stream_chunk_is_dropped(self):
        """During streaming, LiteLLM fires the success hook with an empty
        accumulator before the final chunk arrives. Those calls must NOT
        produce a row."""
        entry = _build_entry(
            {"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            _empty_stream_chunk(),
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 1, 1, 0, 0, 1),
        )
        assert entry is None

    def test_final_stream_chunk_with_usage_is_kept(self):
        entry = _build_entry(
            {"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            _fake_response(content="Done."),
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 1, 1, 0, 0, 1),
        )
        assert entry is not None
        assert entry["completion"] == "Done."
        assert entry["usage"]["total_tokens"] == 13
        assert entry["stream"] is True

    def test_failure_captures_error_type_and_message(self):
        entry = _build_entry(
            {"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            None,
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 1, 1, 0, 0, 1),
            failure=RuntimeError("upstream 503"),
        )
        assert entry is not None
        assert entry["status"] == "failure"
        assert entry["error_type"] == "RuntimeError"
        assert "upstream 503" in entry["error_message"]
        assert entry["completion"] is None
        assert entry["usage"] is None

    def test_dict_response_shape_supported(self):
        """LiteLLM Proxy sometimes passes a dict instead of a ModelResponse."""
        entry = _build_entry(
            {"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            {
                "choices": [{"message": {"content": "From dict."}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                "_hidden_params": {"response_cost": 0.0001},
            },
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 1, 1, 0, 0, 1),
        )
        assert entry is not None
        assert entry["completion"] == "From dict."
        assert entry["usage"]["total_tokens"] == 5
        assert entry["cost_usd"] == pytest.approx(0.0001)

    def test_provider_extracted_from_model_prefix(self):
        for model, expected in [
            ("blockrun/openai/gpt-5.5", "blockrun"),
            ("openai/gpt-4o", "openai"),
            ("just-a-bare-model", None),
            (None, None),
        ]:
            entry = _build_entry(
                {"model": model, "messages": [], "stream": False},
                _fake_response(),
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 1),
            )
            if entry is not None:
                assert entry["provider"] == expected, f"model={model!r}"


# ---------------------------------------------------------------------------
# JSONLLogger writes-to-disk tests
# ---------------------------------------------------------------------------


class TestJSONLLoggerWrites:
    def test_log_success_event_writes_row(self, tmp_path: Path):
        log_path = tmp_path / "calls.jsonl"
        logger = JSONLLogger(log_path)
        logger.log_success_event(
            kwargs={"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            response_obj=_fake_response(),
            start_time=datetime(2026, 1, 1, 0, 0, 0),
            end_time=datetime(2026, 1, 1, 0, 0, 1),
        )
        rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == "success"
        assert rows[0]["completion"] == "Hello world."

    def test_log_failure_event_writes_row(self, tmp_path: Path):
        log_path = tmp_path / "calls.jsonl"
        logger = JSONLLogger(log_path)
        logger.log_failure_event(
            kwargs={"model": "blockrun/openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            response_obj=RuntimeError("upstream 503"),
            start_time=datetime(2026, 1, 1, 0, 0, 0),
            end_time=datetime(2026, 1, 1, 0, 0, 1),
        )
        rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == "failure"
        assert rows[0]["error_type"] == "RuntimeError"

    def test_intermediate_chunks_dont_write(self, tmp_path: Path):
        log_path = tmp_path / "calls.jsonl"
        logger = JSONLLogger(log_path)

        # Three intermediate fires + one final fire
        kwargs = {"model": "blockrun/openai/gpt-5.5", "messages": [], "stream": True}
        start, end = datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 1)
        for _ in range(3):
            logger.log_success_event(kwargs, _empty_stream_chunk(), start, end)
        logger.log_success_event(kwargs, _fake_response(), start, end)

        rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(rows) == 1, f"intermediate fires leaked: got {len(rows)} rows"
        assert rows[0]["completion"] == "Hello world."

    @pytest.mark.asyncio
    async def test_async_hooks_delegate_to_sync(self, tmp_path: Path):
        log_path = tmp_path / "calls.jsonl"
        logger = JSONLLogger(log_path)

        await logger.async_log_success_event(
            kwargs={"model": "blockrun/openai/gpt-5.5", "messages": [], "stream": False},
            response_obj=_fake_response(content="Async ok."),
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1, 0, 0, 1),
        )
        await logger.async_log_failure_event(
            kwargs={"model": "blockrun/openai/gpt-5.5", "messages": [], "stream": True},
            response_obj=ValueError("boom"),
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1, 0, 0, 1),
        )

        rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(rows) == 2
        assert rows[0]["status"] == "success"
        assert rows[0]["completion"] == "Async ok."
        assert rows[1]["status"] == "failure"
        assert rows[1]["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# enable_local_logging integration
# ---------------------------------------------------------------------------


class TestEnableLocalLogging:
    def test_registers_one_logger_per_path(self, tmp_path: Path, monkeypatch):
        import litellm

        # Reset litellm callbacks so the test is hermetic.
        monkeypatch.setattr(litellm, "callbacks", [])

        log_path = tmp_path / "calls.jsonl"
        a = enable_local_logging(log_path)
        b = enable_local_logging(log_path)  # idempotent
        assert a == b == log_path

        loggers = [cb for cb in litellm.callbacks if isinstance(cb, JSONLLogger)]
        assert len(loggers) == 1
        assert loggers[0].path == log_path

    def test_env_var_resolves_default_path(self, tmp_path: Path, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "callbacks", [])
        target = tmp_path / "from-env.jsonl"
        monkeypatch.setenv("BLOCKRUN_LITELLM_LOG", str(target))

        resolved = enable_local_logging()
        assert resolved == target


# ---------------------------------------------------------------------------
# proxy_logger module-level singleton
# ---------------------------------------------------------------------------


def test_proxy_logger_is_jsonl_logger():
    """Mode B (config.yaml callbacks) imports this module-level instance."""
    assert isinstance(proxy_logger, JSONLLogger)
