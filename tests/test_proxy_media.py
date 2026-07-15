"""HTTP-level tests for the media endpoints (image/video/speech/music/sfx).

Every route shares `_media_endpoint` (semaphore + error mapping + audit
logging), so the negative paths are parametrized across the whole media
surface: invalid JSON → 400, missing required field → 400, PaymentError →
402 carrying the gateway ``details``, APIError → its status (clamped to 502
out of range), and ValueError (SDK request validation) → 400 on every route
— the regression that motivated this file was music-with-lyrics 500ing.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from blockrun_llm.types import APIError, PaymentError

import blockrun_litellm.proxy as proxy

# (route, adapter fn to mock, minimal valid body)
ROUTES = [
    ("/v1/images/generations", "image_generation_async", {"prompt": "a cat"}),
    (
        "/v1/images/edits",
        "image_edit_async",
        {"prompt": "make it green", "image": "data:image/png;base64,AA=="},
    ),
    ("/v1/videos/generations", "video_generation_async", {"prompt": "a cat"}),
    ("/v1/audio/speech", "speech_generation_async", {"input": "hi"}),
    ("/v1/audio/generations", "music_generation_async", {"prompt": "lo-fi"}),
    ("/v1/audio/sound-effects", "sound_effect_async", {"text": "boom"}),
]

OK_RESULT: Dict[str, Any] = {
    "created": 1_700_000_000,
    "model": "stub-model",
    "data": [{"url": "https://cdn.blockrun.ai/x"}],
    "txHash": "0xdeadbeef",
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(proxy.app)


def _mock_adapter(monkeypatch: pytest.MonkeyPatch, fn: str, **kwargs: Any) -> AsyncMock:
    mock = AsyncMock(**kwargs)
    monkeypatch.setattr(proxy._adapter, fn, mock)
    return mock


class TestValidation:
    @pytest.mark.parametrize("route,fn,body", ROUTES)
    def test_invalid_json_400(self, client, route, fn, body):
        r = client.post(route, content=b"not-json", headers={"content-type": "application/json"})
        assert r.status_code == 400
        assert r.json()["detail"] == "Invalid JSON body"

    @pytest.mark.parametrize("route,fn,body", ROUTES)
    def test_missing_required_field_400(self, client, route, fn, body):
        r = client.post(route, json={"model": "m"})
        assert r.status_code == 400

    def test_speech_empty_input_400(self, client):
        assert client.post("/v1/audio/speech", json={"input": ""}).status_code == 400

    def test_image_non_integer_n_400(self, client):
        r = client.post("/v1/images/generations", json={"prompt": "a cat", "n": "lots"})
        assert r.status_code == 400
        assert "`n` must be an integer" in r.json()["detail"]


class TestSuccess:
    @pytest.mark.parametrize("route,fn,body", ROUTES)
    def test_ok_result_and_settlement_header(self, client, monkeypatch, route, fn, body):
        _mock_adapter(monkeypatch, fn, return_value=dict(OK_RESULT))
        r = client.post(route, json=body)
        assert r.status_code == 200
        assert r.json() == OK_RESULT
        # In-body txHash surfaces as the settlement header for spend tracking.
        assert "0xdeadbeef" in r.headers["x-blockrun-settlement"]

    def test_no_settlement_header_without_txhash(self, client, monkeypatch):
        result = {k: v for k, v in OK_RESULT.items() if k != "txHash"}
        _mock_adapter(monkeypatch, "speech_generation_async", return_value=result)
        r = client.post("/v1/audio/speech", json={"input": "hi"})
        assert r.status_code == 200
        assert "x-blockrun-settlement" not in r.headers


class TestErrorMapping:
    @pytest.mark.parametrize("route,fn,body", ROUTES)
    def test_payment_error_402_with_details(self, client, monkeypatch, route, fn, body):
        exc = PaymentError(
            "Payment rejected by gateway: transaction_simulation_failed",
            status_code=402,
            response={
                "message": "Payment settlement failed",
                "details": "transaction_simulation_failed",
            },
        )
        _mock_adapter(monkeypatch, fn, side_effect=exc)
        r = client.post(route, json=body)
        assert r.status_code == 402
        assert r.json()["details"] == "transaction_simulation_failed"

    @pytest.mark.parametrize("route,fn,body", ROUTES)
    @pytest.mark.parametrize("code,expected", [(429, 429), (501, 501), (504, 504), (0, 502)])
    def test_api_error_status_clamped(self, client, monkeypatch, route, fn, body, code, expected):
        _mock_adapter(monkeypatch, fn, side_effect=APIError("boom", code))
        r = client.post(route, json=body)
        assert r.status_code == expected
        assert r.json()["error"] == "boom"

    @pytest.mark.parametrize("route,fn,body", ROUTES)
    def test_value_error_400_on_every_route(self, client, monkeypatch, route, fn, body):
        """Regression: SDK request validation (e.g. lyrics + instrumental=True,
        mutually-exclusive video image params) must 400, never 500 — only the
        video route caught ValueError when the endpoints first shipped."""
        _mock_adapter(
            monkeypatch,
            fn,
            side_effect=ValueError("Cannot specify lyrics when instrumental is True"),
        )
        r = client.post(route, json=body)
        assert r.status_code == 400
        assert "lyrics" in r.json()["detail"]


class TestArgumentForwarding:
    @pytest.mark.parametrize(
        "body,expected_text",
        [
            ({"input": "a", "prompt": "b", "text": "c"}, "a"),
            ({"prompt": "b", "text": "c"}, "b"),
            ({"text": "c"}, "c"),
        ],
    )
    def test_speech_alias_precedence(self, client, monkeypatch, body, expected_text):
        mock = _mock_adapter(monkeypatch, "speech_generation_async", return_value=dict(OK_RESULT))
        assert client.post("/v1/audio/speech", json=body).status_code == 200
        assert mock.call_args.kwargs["input"] == expected_text

    def test_music_instrumental_false_and_lyrics_forwarded(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "music_generation_async", return_value=dict(OK_RESULT))
        client.post(
            "/v1/audio/generations",
            json={"prompt": "pop", "instrumental": False, "lyrics": "la la"},
        )
        assert mock.call_args.kwargs["instrumental"] is False
        assert mock.call_args.kwargs["lyrics"] == "la la"

    def test_video_param_filtering(self, client, monkeypatch):
        """False must be forwarded (generate_audio/watermark are real toggles);
        None/absent must be dropped; unknown keys never reach the adapter."""
        mock = _mock_adapter(monkeypatch, "video_generation_async", return_value=dict(OK_RESULT))
        client.post(
            "/v1/videos/generations",
            json={
                "prompt": "a cat",
                "generate_audio": False,
                "seed": None,
                "junk_key": "ignored",
                "duration_seconds": 5,
                "reference_videos": [{"url": "https://example.com/ref.mp4"}],
                "input_type": "reference",
            },
        )
        kwargs = mock.call_args.kwargs
        assert kwargs["generate_audio"] is False
        assert kwargs["duration_seconds"] == 5
        assert kwargs["reference_videos"][0]["url"].endswith("ref.mp4")
        assert kwargs["input_type"] == "reference"
        assert "seed" not in kwargs
        assert "junk_key" not in kwargs
        assert kwargs["model"] is None  # omitted model forwards as None

    def test_image_quality_forwarded(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cat",
                "model": "openai/gpt-image-2",
                "quality": "high",
            },
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["quality"] == "high"

    def test_image_edit_json_multi_image_forwarded(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        images = ["data:image/png;base64,AA==", "data:image/png;base64,AQ=="]
        response = client.post(
            "/v1/images/edits",
            json={
                "prompt": "combine",
                "model": "openai/gpt-image-2",
                "image": images,
                "quality": "medium",
            },
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["image"] == images
        assert mock.call_args.kwargs["quality"] == "medium"

    def test_image_edit_multipart_forwarded(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "combine", "model": "google/nano-banana"},
            files=[
                ("image", ("a.png", b"first", "image/png")),
                ("image", ("b.png", b"second", "image/png")),
            ],
        )
        assert response.status_code == 200
        images = mock.call_args.kwargs["image"]
        assert len(images) == 2
        assert all(value.startswith("data:image/png;base64,") for value in images)
