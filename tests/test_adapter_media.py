"""Adapter-level tests for the media wrappers (video/music/speech/sfx).

Covers the dispatch rules that live below the proxy: Base clients use
``.generate``/``.sound_effect`` while Solana routes through the unified
``SolanaLLMClient`` method names, ``timeout`` is Base-stripped but
Solana-forwarded, client-supplied budgets are clamped to the server cap,
old Solana SDKs degrade to a 501 APIError instead of an AttributeError 500,
and the await ceiling converts a wedged worker into a 504.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from blockrun_llm.types import APIError

from blockrun_litellm import _adapter


class FakeResponse:
    def __init__(self, payload: Dict[str, Any] | None = None) -> None:
        self._payload = payload or {"created": 1, "model": "m", "data": [{"url": "u"}]}

    def model_dump(self, exclude_none: bool = True) -> Dict[str, Any]:
        return self._payload


class FakeBaseVideoClient:
    """Base-shaped: media call is .generate, no .video method."""

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def generate(self, prompt: str, *, model: Any = None, **kw: Any) -> FakeResponse:
        self.calls.append({"prompt": prompt, "model": model, **kw})
        return FakeResponse()


class FakeSolanaMediaClient:
    """Solana-shaped: unified client exposing .video/.music/.speech/.sound_effect."""

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def video(self, prompt: str, *, model: Any = None, **kw: Any) -> FakeResponse:
        self.calls.append({"method": "video", "prompt": prompt, "model": model, **kw})
        return FakeResponse()

    def music(self, prompt: str, **kw: Any) -> FakeResponse:
        self.calls.append({"method": "music", "prompt": prompt, **kw})
        return FakeResponse()

    def speech(self, input: str, **kw: Any) -> FakeResponse:
        self.calls.append({"method": "speech", "input": input, **kw})
        return FakeResponse()

    def sound_effect(self, text: str, **kw: Any) -> FakeResponse:
        self.calls.append({"method": "sound_effect", "text": text, **kw})
        return FakeResponse()


class FakeOldSolanaClient:
    """Pre-#16 SolanaLLMClient: no media methods at all."""


def _route_to(monkeypatch: pytest.MonkeyPatch, client: Any, *, solana: bool) -> None:
    for getter in ("get_video_client", "get_music_client", "get_speech_client"):
        monkeypatch.setattr(_adapter, getter, lambda **_: client)
    monkeypatch.setattr(_adapter, "_is_solana_client", lambda c: solana and c is client)


class TestVideoDispatch:
    @pytest.mark.asyncio
    async def test_bare_seedance_model_is_canonicalized_for_solana(self, monkeypatch):
        client = FakeSolanaMediaClient()
        _route_to(monkeypatch, client, solana=True)
        await _adapter.video_generation_async("a cat", model="seedance-2.0-fast")
        (call,) = client.calls
        assert call["model"] == "bytedance/seedance-2.0-fast"

    @pytest.mark.asyncio
    async def test_bare_seedance_model_is_canonicalized_for_base(self, monkeypatch):
        # Canonicalization runs above the chain split, so Base gets it too —
        # both catalogs name Seedance the same way.
        client = FakeBaseVideoClient()
        _route_to(monkeypatch, client, solana=False)
        await _adapter.video_generation_async("a cat", model="seedance-2.0-fast")
        (call,) = client.calls
        assert call["model"] == "bytedance/seedance-2.0-fast"

    @pytest.mark.asyncio
    async def test_mixed_case_seedance_model_is_lowercased(self, monkeypatch):
        # The catalog lookup is an exact match, so prefixing alone isn't enough.
        client = FakeSolanaMediaClient()
        _route_to(monkeypatch, client, solana=True)
        await _adapter.video_generation_async("a cat", model="Seedance-2.0-Fast")
        (call,) = client.calls
        assert call["model"] == "bytedance/seedance-2.0-fast"

    @pytest.mark.asyncio
    async def test_non_seedance_model_passes_through_untouched(self, monkeypatch):
        client = FakeBaseVideoClient()
        _route_to(monkeypatch, client, solana=False)
        await _adapter.video_generation_async("a cat", model="xai/grok-imagine-video")
        (call,) = client.calls
        assert call["model"] == "xai/grok-imagine-video"

    @pytest.mark.asyncio
    async def test_bare_grok_model_is_canonicalized(self, monkeypatch):
        client = FakeBaseVideoClient()
        _route_to(monkeypatch, client, solana=False)
        await _adapter.video_generation_async("a cat", model="grok-imagine-video")
        (call,) = client.calls
        assert call["model"] == "xai/grok-imagine-video"

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("seedance-2.0-fast", "bytedance/seedance-2.0-fast"),
            ("  seedance-2.0-fast\n", "bytedance/seedance-2.0-fast"),
            ("GROK-IMAGINE-VIDEO", "xai/grok-imagine-video"),
            ("bytedance/seedance-2.0", "bytedance/seedance-2.0"),
            # Ambiguous across vendors (azure/sora-2 vs openai/sora-2) — must NOT
            # be guessed; it reaches the gateway bare and 400s with the real list.
            ("sora-2", "sora-2"),
            ("nonsense", "nonsense"),
            (None, None),
            (123, 123),
        ],
    )
    def test_canonical_video_model_table(self, given, expected):
        assert _adapter._canonical_video_model(given) == expected

    @pytest.mark.asyncio
    async def test_base_strips_timeout_and_drops_none(self, monkeypatch):
        client = FakeBaseVideoClient()
        _route_to(monkeypatch, client, solana=False)
        out = await _adapter.video_generation_async(
            "a cat", model="xai/grok-imagine-video", duration_seconds=5, timeout=600, seed=None
        )
        assert out == FakeResponse().model_dump()
        (call,) = client.calls
        assert "timeout" not in call  # Base VideoClient.generate has no timeout kwarg
        assert "seed" not in call  # None params filtered
        assert call["duration_seconds"] == 5

    @pytest.mark.asyncio
    async def test_solana_uses_video_method_and_keeps_timeout(self, monkeypatch):
        client = FakeSolanaMediaClient()
        _route_to(monkeypatch, client, solana=True)
        await _adapter.video_generation_async("a cat", timeout=600)
        (call,) = client.calls
        assert call["method"] == "video"
        assert call["timeout"] == 600

    @pytest.mark.asyncio
    async def test_budget_and_timeout_clamped_to_cap(self, monkeypatch):
        client = FakeSolanaMediaClient()
        _route_to(monkeypatch, client, solana=True)
        await _adapter.video_generation_async("a cat", budget_seconds=86400, timeout="7200")
        (call,) = client.calls
        assert call["budget_seconds"] == _adapter._VIDEO_BUDGET_CAP_S
        assert call["timeout"] == _adapter._VIDEO_BUDGET_CAP_S  # str coerced then clamped

    @pytest.mark.asyncio
    async def test_non_numeric_budget_raises_value_error(self, monkeypatch):
        client = FakeSolanaMediaClient()
        _route_to(monkeypatch, client, solana=True)
        with pytest.raises(ValueError):
            await _adapter.video_generation_async("a cat", budget_seconds="a while")


class TestSolanaSdkGuard:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            lambda: _adapter.video_generation_async("p"),
            lambda: _adapter.music_generation_async("p"),
            lambda: _adapter.speech_generation_async("p"),
            lambda: _adapter.sound_effect_async("p"),
        ],
    )
    async def test_old_solana_sdk_degrades_to_501(self, monkeypatch, call):
        client = FakeOldSolanaClient()
        _route_to(monkeypatch, client, solana=True)
        with pytest.raises(APIError) as exc_info:
            await call()
        assert exc_info.value.status_code == 501
        assert "blockrun-llm" in str(exc_info.value)


class TestOtherMediaDispatch:
    @pytest.mark.asyncio
    async def test_music_value_error_propagates(self, monkeypatch):
        class RejectingClient:
            def generate(self, prompt, **kw):
                raise ValueError("Cannot specify lyrics when instrumental is True")

        _route_to(monkeypatch, RejectingClient(), solana=False)
        with pytest.raises(ValueError, match="lyrics"):
            await _adapter.music_generation_async("pop", lyrics="la", instrumental=True)

    @pytest.mark.asyncio
    async def test_speech_solana_dispatch_drops_none(self, monkeypatch):
        client = FakeSolanaMediaClient()
        _route_to(monkeypatch, client, solana=True)
        await _adapter.speech_generation_async("hi", voice="sarah", speed=None)
        (call,) = client.calls
        assert call["method"] == "speech"
        assert call["voice"] == "sarah"
        assert "speed" not in call

    @pytest.mark.asyncio
    async def test_sound_effect_base_uses_sound_effect(self, monkeypatch):
        class FakeBaseSpeechClient(FakeSolanaMediaClient):
            pass  # Base SpeechClient also exposes .sound_effect — same shape

        client = FakeBaseSpeechClient()
        _route_to(monkeypatch, client, solana=False)
        await _adapter.sound_effect_async("boom", duration_seconds=2.0)
        (call,) = client.calls
        assert call["method"] == "sound_effect"
        assert call["duration_seconds"] == 2.0


class TestClientRoutingAndCache:
    def test_base_clients_route_and_cache(self, monkeypatch):
        monkeypatch.setattr(_adapter, "_media_clients", {})
        from blockrun_llm import MusicClient, SpeechClient, VideoClient

        url = "https://blockrun.ai/api"
        pk = "0x" + "11" * 32
        for getter, cls in [
            (_adapter.get_video_client, VideoClient),
            (_adapter.get_music_client, MusicClient),
            (_adapter.get_speech_client, SpeechClient),
        ]:
            c1 = getter(api_url=url, private_key=pk)
            c2 = getter(api_url=url, private_key=pk)
            assert isinstance(c1, cls)
            assert c1 is c2  # cache hit

    def test_solana_url_reuses_image_client(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(_adapter, "get_image_client", lambda **_: sentinel)
        assert _adapter.get_video_client(api_url="https://sol.blockrun.ai/api") is sentinel
        assert _adapter.get_music_client(api_url="https://sol.blockrun.ai/api") is sentinel
        assert _adapter.get_speech_client(api_url="https://sol.blockrun.ai/api") is sentinel


class TestAwaitCeiling:
    @pytest.mark.asyncio
    async def test_wedged_worker_becomes_504(self):
        import time as _time

        with pytest.raises(APIError) as exc_info:
            await _adapter._run_media(lambda: _time.sleep(2), ceiling=0.05)
        assert exc_info.value.status_code == 504
        assert "ceiling" in str(exc_info.value)
