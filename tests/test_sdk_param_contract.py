"""Contract tests: every param the adapter forwards must exist on the real SDK.

The rest of the suite fakes the SDK clients, so a kwarg the adapter invents is
invisible there — the fake's ``**kwargs`` swallows it and the assertion passes.
The real clients have *closed* signatures (``ImageClient.generate`` even raises
TypeError on unknown kwargs), so a param that no release accepts is a runtime
failure on the first real call.

These tests bind the adapter's forwarded params against the installed SDK's
actual signatures, which is the only place that mismatch shows up before
production.
"""

import inspect

import pytest

from blockrun_litellm import _adapter

blockrun_image = pytest.importorskip("blockrun_llm.image")
blockrun_video = pytest.importorskip("blockrun_llm.video")


def _accepted(func) -> set:
    """Param names a function accepts, or None if it takes **kwargs."""
    params = inspect.signature(func).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name != "self"
    }


def _solana_client():
    try:
        from blockrun_llm.solana_client import SolanaLLMClient
    except Exception:
        pytest.skip("blockrun-llm[solana] not installed")
    return SolanaLLMClient


class TestVideoParamContract:
    def test_base_video_accepts_every_forwarded_key(self):
        accepted = _accepted(blockrun_video.VideoClient.generate)
        if accepted is None:
            pytest.skip("VideoClient.generate takes **kwargs")
        # timeout is Solana-only; the adapter pops it before calling Base.
        forwarded = set(_adapter.VIDEO_PARAM_KEYS) - {"timeout"}
        assert forwarded <= accepted, (
            f"VIDEO_PARAM_KEYS forwards params VideoClient.generate rejects: "
            f"{sorted(forwarded - accepted)}"
        )

    def test_solana_video_accepts_every_forwarded_key(self):
        accepted = _accepted(_solana_client().video)
        if accepted is None:
            pytest.skip("SolanaLLMClient.video takes **kwargs")
        forwarded = set(_adapter.VIDEO_PARAM_KEYS)
        assert forwarded <= accepted, (
            f"VIDEO_PARAM_KEYS forwards params SolanaLLMClient.video rejects: "
            f"{sorted(forwarded - accepted)}"
        )


class TestImageParamContract:
    """The adapter's image kwargs are explicit, so assert the signatures directly."""

    def test_base_image_generate_accepts_forwarded_params(self):
        accepted = _accepted(blockrun_image.ImageClient.generate)
        if accepted is not None:
            assert {"prompt", "model", "size", "n"} <= accepted

    def test_solana_image_accepts_forwarded_params(self):
        accepted = _accepted(_solana_client().image)
        if accepted is not None:
            assert {"prompt", "model", "size", "n"} <= accepted

    def test_base_image_edit_accepts_forwarded_params(self):
        accepted = _accepted(blockrun_image.ImageClient.edit)
        if accepted is not None:
            assert {"prompt", "image", "model", "mask", "size", "n"} <= accepted

    def test_solana_image_edit_accepts_forwarded_params(self):
        accepted = _accepted(_solana_client().image_edit)
        if accepted is not None:
            assert {"prompt", "image", "model", "mask", "size", "n"} <= accepted
