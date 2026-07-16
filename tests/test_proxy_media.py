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

# The image routes deliberately keep `_optional_str` semantics for `model`
# (blank → unset → gateway default). Only the routes below treat a blank `model`
# as a caller bug — see `_require_named_model`.
MODEL_NAMED_ROUTES = [r for r in ROUTES if not r[0].startswith("/v1/images/")]

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

    @pytest.mark.parametrize("route,fn,body", MODEL_NAMED_ROUTES)
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_model_400_and_never_dispatched(self, client, monkeypatch, route, fn, body, blank):
        # The money assertion is `assert_not_awaited`: a blank model is falsy, so
        # the SDK's `model or DEFAULT_MODEL` would silently bill the default.
        mock = _mock_adapter(monkeypatch, fn, return_value=dict(OK_RESULT))
        r = client.post(route, json={**body, "model": blank})
        assert r.status_code == 400
        assert "`model` must not be empty" in r.json()["detail"]
        mock.assert_not_awaited()

    @pytest.mark.parametrize("route,fn,body", MODEL_NAMED_ROUTES)
    def test_non_string_model_400_and_never_dispatched(self, client, monkeypatch, route, fn, body):
        mock = _mock_adapter(monkeypatch, fn, return_value=dict(OK_RESULT))
        r = client.post(route, json={**body, "model": 123})
        assert r.status_code == 400
        assert "`model` must be a string" in r.json()["detail"]
        mock.assert_not_awaited()

    @pytest.mark.parametrize("route,fn,body", MODEL_NAMED_ROUTES)
    def test_omitted_model_still_opts_into_the_default(self, client, monkeypatch, route, fn, body):
        # Omitting `model` stays documented and free — only a *blank* one is a bug.
        mock = _mock_adapter(monkeypatch, fn, return_value=dict(OK_RESULT))
        r = client.post(route, json=body)
        assert r.status_code == 200
        assert mock.await_args.kwargs["model"] is None


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
            },
        )
        kwargs = mock.call_args.kwargs
        assert kwargs["generate_audio"] is False
        assert kwargs["duration_seconds"] == 5
        assert "seed" not in kwargs
        assert "junk_key" not in kwargs
        assert kwargs["model"] is None  # omitted model forwards as None

    def test_video_input_type_forwarded(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "video_generation_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/videos/generations",
            json={
                "prompt": "the portrait turns",
                "image_url": "https://example.com/a.jpg",
                "input_type": "image",
            },
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["input_type"] == "image"

    def test_image_quality_forwarded_on_solana(self, client, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_API_URL", "https://sol.blockrun.ai/api")
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "a cat", "model": "openai/gpt-image-2", "quality": "low"},
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["quality"] == "low"
        assert "x-blockrun-warning" not in response.headers

    def test_image_edit_quality_forwarded_on_solana(self, client, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_API_URL", "https://sol.blockrun.ai/api")
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            json={
                "prompt": "make it green",
                "image": "data:image/png;base64,AA==",
                "quality": "high",
            },
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["quality"] == "high"

    def test_image_quality_on_base_is_served_dropped_and_warned(self, client, monkeypatch):
        """0.6.1 returned 200 and ignored quality; 0.7.0 broke that with a 400.

        Restore the 200 — quality is a standard OpenAI Images param and this is
        a drop-in-compatible route — but surface the drop in a header so it
        isn't the silent no-op 0.6.1 had.
        """
        monkeypatch.delenv("BLOCKRUN_API_URL", raising=False)  # Base
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "a cat", "model": "openai/dall-e-3", "quality": "hd"},
        )
        assert response.status_code == 200, "a Base caller's working 0.6.1 code must not break"
        assert mock.call_args.kwargs["quality"] is None, "must not reach the Base SDK"
        assert "Solana only" in response.headers.get("x-blockrun-warning", "")

    def test_image_edit_quality_on_base_is_served_dropped_and_warned(self, client, monkeypatch):
        monkeypatch.delenv("BLOCKRUN_API_URL", raising=False)
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            json={
                "prompt": "make it green",
                "image": "data:image/png;base64,AA==",
                "quality": "hd",
            },
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["quality"] is None
        assert "Solana only" in response.headers.get("x-blockrun-warning", "")

    def test_image_edit_json_multi_image_forwarded(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        images = ["data:image/png;base64,AA==", "data:image/png;base64,AQ=="]
        response = client.post(
            "/v1/images/edits",
            json={
                "prompt": "combine",
                "model": "openai/gpt-image-2",
                "image": images,
            },
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["image"] == images
        assert mock.call_args.kwargs["model"] == "openai/gpt-image-2"

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

    def test_blank_quality_is_treated_as_unset(self, client, monkeypatch):
        """A blank field means "not set", not "set to empty".

        Multipart clients routinely emit every optional field, blank ones
        included. Without normalization a blank `quality` would reach the
        Solana-only guard and 400 about a parameter the caller never sent.
        """
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        for blank in ("", "   "):
            response = client.post(
                "/v1/images/generations",
                json={"prompt": "a cat", "quality": blank},
            )
            assert response.status_code == 200
            assert mock.call_args.kwargs["quality"] is None

    def test_blank_quality_multipart_is_treated_as_unset(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            files={"image": ("a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png")},
            data={"prompt": "make it green", "quality": ""},
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["quality"] is None

    def test_prompt_sent_as_file_part_is_rejected(self, client, monkeypatch):
        """A multipart `prompt` file part is truthy but is not text.

        Before the isinstance guard, str() on the UploadFile billed
        "UploadFile(filename='p.txt', ...)" as the generation prompt — a paid
        call against a garbage prompt, and a 200 hiding it.
        """
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            files={
                "image": ("a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png"),
                "prompt": ("p.txt", b"hello", "text/plain"),
            },
        )
        assert response.status_code == 400
        assert "prompt" in str(response.json()).lower()
        mock.assert_not_called()  # nothing billable escaped

    def test_blank_prompt_is_rejected(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        for bad in ("", "   ", 123, None):
            response = client.post("/v1/images/generations", json={"prompt": bad})
            assert response.status_code == 400, f"{bad!r} should be refused"
        mock.assert_not_called()

    def test_non_string_image_payload_is_rejected(self, client, monkeypatch):
        """Refuse locally instead of paying for the gateway to say no."""
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        for bad in (12345, {"url": "x"}, [123], ""):
            response = client.post(
                "/v1/images/edits",
                json={"prompt": "make it green", "image": bad},
            )
            assert response.status_code == 400, f"{bad!r} should be refused"
        mock.assert_not_called()

    def test_non_string_mask_is_rejected(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            json={
                "prompt": "make it green",
                "image": "data:image/png;base64,AA==",
                "mask": 123,
            },
        )
        assert response.status_code == 400
        mock.assert_not_called()

    def test_valid_multipart_edit_still_works(self, client, monkeypatch):
        """The guards must not break the happy path they surround."""
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            files={"image": ("a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png")},
            data={"prompt": "make it green", "model": "openai/gpt-image-2"},
        )
        assert response.status_code == 200
        kwargs = mock.call_args.kwargs
        assert kwargs["prompt"] == "make it green"
        assert kwargs["image"].startswith("data:image/png;base64,")

    def test_oversized_upload_is_rejected(self, client, monkeypatch):
        """Fail fast rather than buffer + base64-inflate a huge body first."""
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (13 * 1024 * 1024)
        response = client.post(
            "/v1/images/edits",
            files={"image": ("big.png", big, "image/png")},
            data={"prompt": "make it green"},
        )
        assert response.status_code == 413
        mock.assert_not_called()

    def test_too_many_image_parts_rejected(self, client, monkeypatch):
        """A buffering guard only — per-model caps stay the gateway's call."""
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        parts = [("image", (f"{i}.png", b"\x89PNG\r\n\x1a\n", "image/png")) for i in range(20)]
        response = client.post("/v1/images/edits", files=parts, data={"prompt": "green"})
        assert response.status_code == 400
        assert "image parts" in str(response.json())
        mock.assert_not_called()

    def test_multi_image_under_the_cap_still_fuses(self, client, monkeypatch):
        """The cap must not disturb the multi-image fusion the PR exists for."""
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        parts = [("image", (f"{i}.png", b"\x89PNG\r\n\x1a\n", "image/png")) for i in range(3)]
        response = client.post("/v1/images/edits", files=parts, data={"prompt": "fuse"})
        assert response.status_code == 200
        assert len(mock.call_args.kwargs["image"]) == 3

    def test_bad_typed_size_or_model_is_rejected_not_silently_defaulted(self, client, monkeypatch):
        """Refusing is free; guessing bills the caller for the wrong image.

        0.7.0 coerced a wrong-typed value to None on /v1/images/edits, so the
        SDK substituted its default and charged: {"size": 512} meaning 512x512
        quietly rendered 1024x1024 at the default model.
        """
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        for field, bad in (("size", 512), ("model", 99), ("quality", 5)):
            body = {"prompt": "g", "image": "data:image/png;base64,AA==", field: bad}
            response = client.post("/v1/images/edits", json=body)
            assert response.status_code == 400, f"{field}={bad!r} must not be guessed at"
        mock.assert_not_called()

    def test_n_is_bounded_before_it_can_cost_anything(self, client, monkeypatch):
        """Base's image2image gateway schema has no int/bounds on n, so n=1000
        passes zod, takes payment, then 400s at the provider — prepaid USDC gone.
        """
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        for bad in (0, -5, 1000, 11):
            response = client.post("/v1/images/generations", json={"prompt": "a", "n": bad})
            assert response.status_code == 400, f"n={bad} should be refused locally"
        mock.assert_not_called()

    def test_n_within_bounds_still_works(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        response = client.post("/v1/images/generations", json={"prompt": "a", "n": 10})
        assert response.status_code == 200
        assert mock.call_args.kwargs["n"] == 10

    def test_blank_optional_strings_are_unset_not_empty(self, client, monkeypatch):
        """Blank size/model reached the SDK as "" before: Base billed a default
        image, Solana 400'd. Same input, different chain, neither intended.
        """
        mock = _mock_adapter(monkeypatch, "image_generation_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/generations", json={"prompt": "a", "size": "", "model": "  "}
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["size"] is None
        assert mock.call_args.kwargs["model"] is None

    def test_blank_multipart_mask_is_unset(self, client, monkeypatch):
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            files={"image": ("a.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"prompt": "green", "mask": "", "size": "", "n": ""},
        )
        assert response.status_code == 200
        assert mock.call_args.kwargs["mask"] is None
        assert mock.call_args.kwargs["size"] is None
        assert mock.call_args.kwargs["n"] == 1

    def test_multipart_image_order_is_wire_order(self, client, monkeypatch):
        """Order is load-bearing for fusion ("the logo from image 2 on image 1").

        getlist() per field name grouped all `image` before all `image[]`,
        silently reordering a client that mixed both spellings.
        """
        mock = _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        png = b"\x89PNG\r\n\x1a\n"
        response = client.post(
            "/v1/images/edits",
            files=[
                ("image[]", ("first.png", png + b"\x01", "image/png")),
                ("image", ("second.png", png + b"\x02", "image/png")),
            ],
            data={"prompt": "fuse"},
        )
        assert response.status_code == 200
        images = mock.call_args.kwargs["image"]
        assert len(images) == 2
        # first.png was sent first, so it must arrive first despite the name mismatch
        import base64
        assert base64.b64decode(images[0].split(",", 1)[1]).endswith(b"\x01")

    def test_urlencoded_body_names_the_real_problem(self, client, monkeypatch):
        _mock_adapter(monkeypatch, "image_edit_async", return_value=dict(OK_RESULT))
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "g", "image": "data:image/png;base64,AA=="},
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 400
        assert "urlencoded" in str(response.json()).lower()
