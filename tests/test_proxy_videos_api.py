"""HTTP-level tests for the OpenAI-compatible Videos API (/v1/videos).

LiteLLM's video generation (litellm.video_generation / proxy /v1/videos)
speaks the OpenAI Videos spec: POST {api_base}/videos returns a video JOB
object immediately, then GET /videos/{id} polls status and GET
/videos/{id}/content downloads bytes. The sidecar's native
/v1/videos/generations (blocking) is invisible to LiteLLM — these routes are
what make `model: openai/xai/grok-imagine-video` + `api_base: <sidecar>`
actually callable from a LiteLLM proxy.

The blocking SDK call runs as a background asyncio task, so tests use the
TestClient as a context manager (keeps one event loop alive across requests)
and poll the status route until the job settles.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from blockrun_llm.types import APIError, PaymentError

import blockrun_litellm.proxy as proxy

VIDEO_RESULT: Dict[str, Any] = {
    "created": 1_700_000_000,
    "model": "xai/grok-imagine-video",
    "data": [{"url": "https://cdn.blockrun.ai/videos/abc.mp4", "duration_seconds": 8}],
    "txHash": "0xfeedbeef",
}


@pytest.fixture
def client():
    # Context manager keeps the app's event loop (and thus background video
    # jobs) alive across requests.
    with TestClient(proxy.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_job_store():
    proxy._video_jobs.clear()
    yield
    proxy._video_jobs.clear()


def _mock_video_adapter(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> AsyncMock:
    mock = AsyncMock(**kwargs)
    monkeypatch.setattr(proxy._adapter, "video_generation_async", mock)
    return mock


def _wait_terminal(client: TestClient, video_id: str, deadline_s: float = 5.0) -> Dict[str, Any]:
    """Poll GET /v1/videos/{id} until the job leaves queued/in_progress."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        r = client.get(f"/v1/videos/{video_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] not in ("queued", "in_progress"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {video_id} never settled: {body}")


class TestCreate:
    def test_returns_video_job_object(self, client, monkeypatch):
        _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))
        r = client.post(
            "/v1/videos",
            json={"model": "xai/grok-imagine-video", "prompt": "a cat surfing", "seconds": "8"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["id"].startswith("video_")
        assert body["object"] == "video"
        assert body["status"] in ("queued", "in_progress")
        assert body["model"] == "xai/grok-imagine-video"
        assert body["seconds"] == "8"

    def test_missing_prompt_400(self, client):
        r = client.post("/v1/videos", json={"model": "m"})
        assert r.status_code == 400

    def test_invalid_json_400(self, client):
        r = client.post(
            "/v1/videos", content=b"not-json", headers={"content-type": "application/json"}
        )
        assert r.status_code == 400

    def test_blank_model_400_before_a_job_is_created(self, client, monkeypatch):
        # This route creates the job and returns 202-style at once, so a blank
        # model must be refused up front — once the background task dispatches,
        # the default model is billed and there's nothing to take back.
        mock = _mock_video_adapter(monkeypatch, return_value={"url": "u"})
        r = client.post("/v1/videos", json={"prompt": "a cat", "model": "  "})
        assert r.status_code == 400
        assert "`model` must not be empty" in r.json()["detail"]
        mock.assert_not_awaited()

    def test_invalid_seconds_400(self, client):
        r = client.post("/v1/videos", json={"prompt": "a cat", "seconds": "lots"})
        assert r.status_code == 400
        assert "seconds" in r.json()["detail"]

    def test_multipart_input_reference_400(self, client):
        """LiteLLM sends multipart only when `input_reference` (a raw image
        file) is passed. BlockRun's gateway takes image URLs, not uploads —
        reject with a clear pointer instead of a parse error."""
        r = client.post(
            "/v1/videos",
            data={"model": "m", "prompt": "p"},
            files={"input_reference": ("cat.png", b"\x89PNG", "image/png")},
        )
        assert r.status_code == 400
        assert "image_url" in r.json()["detail"]

    def test_openai_params_mapped_to_gateway_shape(self, client, monkeypatch):
        mock = _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))
        r = client.post(
            "/v1/videos",
            json={"model": "m", "prompt": "a cat", "seconds": "4", "size": "720x1280"},
        )
        assert r.status_code == 200
        _wait_terminal(client, r.json()["id"])
        kwargs = mock.call_args.kwargs
        assert kwargs["prompt"] == "a cat"
        assert kwargs["model"] == "m"
        assert kwargs["duration_seconds"] == 4
        assert kwargs["resolution"] == "720p"
        assert kwargs["aspect_ratio"] == "9:16"

    def test_native_params_forwarded_for_direct_callers(self, client, monkeypatch):
        mock = _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))
        r = client.post(
            "/v1/videos",
            json={
                "prompt": "a cat",
                "image_url": "https://example.com/cat.png",
                "generate_audio": False,
                "junk_key": "ignored",
            },
        )
        assert r.status_code == 200
        _wait_terminal(client, r.json()["id"])
        kwargs = mock.call_args.kwargs
        assert kwargs["image_url"] == "https://example.com/cat.png"
        assert kwargs["generate_audio"] is False
        assert "junk_key" not in kwargs
        # No seconds passed -> nothing invented for the adapter or the echo.
        assert "duration_seconds" not in kwargs
        assert r.json().get("seconds") is None


class TestStatus:
    def test_unknown_id_404(self, client):
        assert client.get("/v1/videos/video_nope").status_code == 404

    def test_completed_job(self, client, monkeypatch):
        _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))
        r = client.post("/v1/videos", json={"prompt": "a cat", "seconds": "8"})
        body = _wait_terminal(client, r.json()["id"])
        assert body["status"] == "completed"
        assert body["object"] == "video"
        # Real clip duration from the gateway response wins over the request echo.
        assert body["seconds"] == "8"

    def test_completed_settlement_header(self, client, monkeypatch):
        _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))
        r = client.post("/v1/videos", json={"prompt": "a cat"})
        video_id = r.json()["id"]
        _wait_terminal(client, video_id)
        resp = client.get(f"/v1/videos/{video_id}")
        assert "0xfeedbeef" in resp.headers["x-blockrun-settlement"]

    @pytest.mark.parametrize(
        "exc,expected_code",
        [
            (APIError("upstream exploded", 503), "upstream_error"),
            (PaymentError("insufficient balance"), "payment_error"),
            (ValueError("last_frame_url requires image_url"), "invalid_request"),
        ],
    )
    def test_failed_job_surfaces_error(self, client, monkeypatch, exc, expected_code):
        _mock_video_adapter(monkeypatch, side_effect=exc)
        r = client.post("/v1/videos", json={"prompt": "a cat"})
        body = _wait_terminal(client, r.json()["id"])
        assert body["status"] == "failed"
        assert body["error"]["code"] == expected_code
        assert str(exc) in body["error"]["message"]


class TestContent:
    def test_unknown_id_404(self, client):
        assert client.get("/v1/videos/video_nope/content").status_code == 404

    def test_not_ready_409(self, client, monkeypatch):
        gate = asyncio.Event()

        async def _blocked(**kwargs):
            await gate.wait()
            return dict(VIDEO_RESULT)

        monkeypatch.setattr(proxy._adapter, "video_generation_async", _blocked)
        r = client.post("/v1/videos", json={"prompt": "a cat"})
        video_id = r.json()["id"]
        resp = client.get(f"/v1/videos/{video_id}/content")
        assert resp.status_code == 409
        # Unblock so the background task doesn't leak past the test.
        proxy._video_jobs.clear()

    def test_streams_completed_video_bytes(self, client, monkeypatch):
        _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))

        async def _fake_fetch(url: str):
            assert url == "https://cdn.blockrun.ai/videos/abc.mp4"
            return b"MP4DATA", "video/mp4"

        monkeypatch.setattr(proxy, "_fetch_video_content", _fake_fetch)
        r = client.post("/v1/videos", json={"prompt": "a cat"})
        video_id = r.json()["id"]
        _wait_terminal(client, video_id)
        resp = client.get(f"/v1/videos/{video_id}/content")
        assert resp.status_code == 200
        assert resp.content == b"MP4DATA"
        assert resp.headers["content-type"].startswith("video/mp4")

    def test_failed_job_content_409(self, client, monkeypatch):
        _mock_video_adapter(monkeypatch, side_effect=APIError("boom", 503))
        r = client.post("/v1/videos", json={"prompt": "a cat"})
        video_id = r.json()["id"]
        _wait_terminal(client, video_id)
        assert client.get(f"/v1/videos/{video_id}/content").status_code == 409


class TestJobExpiry:
    def test_expired_jobs_pruned_on_create(self, client, monkeypatch):
        _mock_video_adapter(monkeypatch, return_value=dict(VIDEO_RESULT))
        r = client.post("/v1/videos", json={"prompt": "a cat"})
        old_id = r.json()["id"]
        _wait_terminal(client, old_id)
        # Age the job past the TTL, then trigger the prune with a new create.
        proxy._video_jobs[old_id]["created_at"] -= proxy._VIDEO_JOB_TTL_S + 1
        client.post("/v1/videos", json={"prompt": "another cat"})
        assert old_id not in proxy._video_jobs
        assert client.get(f"/v1/videos/{old_id}").status_code == 404
