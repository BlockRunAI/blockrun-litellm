"""Contract tests: every param the adapter forwards must exist on the real SDK.

The rest of the suite fakes the SDK clients, so a kwarg the adapter invents is
invisible there — the fake's ``**kwargs`` swallows it and the assertion passes.
That is how an earlier revision shipped four params no release accepted, with
96 green media tests behind it.

Two client shapes need two techniques, and conflating them is how this file was
itself briefly vacuous:

* **Closed signatures** (``VideoClient.generate``, the Solana clients) name
  every param, so :func:`_accepted` can read them and set-compare.
* **Open signatures** (``ImageClient.generate`` / ``edit``) declare ``**kwargs``
  and validate at *runtime* to produce a friendlier error. Signature inspection
  sees only ``**kwargs`` and learns nothing, so those must be probed by
  **calling** them — see :func:`_rejects`.

Every probe stubs the transport. A contract test must never be able to reach a
gateway and settle real funds.
"""

import inspect

import pytest

from blockrun_litellm import _adapter

blockrun_image = pytest.importorskip("blockrun_llm.image")
blockrun_video = pytest.importorskip("blockrun_llm.video")

_REACHED_TRANSPORT = object()


def _accepted(func) -> set:
    """Param names a function accepts, or None if it takes ``**kwargs``.

    A None result means "signature inspection cannot answer this" — NOT "any
    kwarg is fine". Callers must fall back to :func:`_rejects` rather than skip
    the assertion, or the test silently passes on everything.
    """
    params = inspect.signature(func).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name != "self"
    }


class _StubReached(Exception):
    """The call got past the runtime guard and hit the stubbed transport."""


def _rejects(monkeypatch, method_name: str, *args, **kwargs) -> bool:
    """Does ``ImageClient.<method_name>`` refuse these kwargs before the wire?

    True when the runtime guard raises TypeError; False when the call reaches
    the stubbed transport, which means the SDK now accepts the kwarg.

    Every other exception propagates, deliberately. An earlier version ended in
    ``except Exception: return False``, which made the "accepts" assertions
    vacuous: delete ``ImageClient.edit`` and the AttributeError was swallowed
    into False, so ``assert not _rejects(...)`` passed green while every Base
    edit call would 500. This file exists to catch tests that cannot fail —
    it does not get to contain one.
    """
    ImageClient = blockrun_image.ImageClient
    assert hasattr(ImageClient, method_name), (
        f"ImageClient.{method_name} no longer exists — the adapter's Base branch "
        f"calls it and would raise AttributeError at runtime"
    )

    def _stub(self, *a, **k):
        raise _StubReached()

    # raising=True: if the SDK renames its transport method this must fail loudly
    # rather than silently stub nothing and let a probe hit the real network.
    monkeypatch.setattr(ImageClient, "_request_with_payment", _stub, raising=True)
    client = ImageClient.__new__(ImageClient)  # the guard runs before any I/O
    try:
        getattr(client, method_name)(*args, **kwargs)
    except TypeError:
        return True
    except _StubReached:
        return False
    return False


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


_DATA_URI = "data:image/png;base64,AA=="


class TestSolanaImageParamContract:
    """Closed signatures — read them directly."""

    def test_solana_image_accepts_forwarded_params(self):
        accepted = _accepted(_solana_client().image)
        assert accepted is not None, "SolanaLLMClient.image gained **kwargs — probe it instead"
        assert {"prompt", "model", "size", "n", "quality"} <= accepted

    def test_solana_image_edit_accepts_forwarded_params(self):
        accepted = _accepted(_solana_client().image_edit)
        assert accepted is not None, "SolanaLLMClient.image_edit gained **kwargs — probe it instead"
        assert {"prompt", "image", "model", "mask", "size", "n", "quality"} <= accepted


class TestBaseImageParamContract:
    """Open signatures — ``ImageClient`` declares ``**kwargs`` and validates at
    runtime, so these must be probed by calling, not by reading the signature.

    Reading it was the earlier mistake here: :func:`_accepted` returns None for
    ``**kwargs``, so a guard written as ``if accepted is not None: assert ...``
    never ran. The file that exists to catch vacuous fakes was vacuous the same
    way.
    """

    def test_base_image_generate_accepts_forwarded_params(self, monkeypatch):
        """What the adapter sends on the Base branch must get past the guard."""
        assert not _rejects(
            monkeypatch, "generate", "a cat", model="google/nano-banana", size="1024x1024", n=1
        ), "ImageClient.generate rejects a param the adapter forwards"

    def test_base_image_edit_accepts_forwarded_params(self, monkeypatch):
        assert not _rejects(
            monkeypatch,
            "edit",
            "make it green",
            _DATA_URI,
            model="openai/gpt-image-2",
            mask=None,
            size="1024x1024",
            n=1,
        ), "ImageClient.edit rejects a param the adapter forwards"

    def test_base_image_client_still_rejects_quality(self, monkeypatch):
        """The Solana-only asymmetry is load-bearing, not an oversight.

        The Base gateway has no quality field and strips unknown keys, so a
        value routed there would vanish silently. ImageClient refuses it, and
        the adapter's Base branch 400s before reaching the SDK at all.

        If a future SDK teaches ImageClient about quality, this fails — and it
        fails for the realistic evolution too, where quality is added as a named
        param *while keeping* ``**kwargs`` for the friendly error. A signature
        check could not see that; calling the method can.
        """
        assert _rejects(
            monkeypatch, "generate", "a cat", quality="low"
        ), "ImageClient.generate now accepts quality — revisit the adapter's Base-branch guard"
        assert _rejects(
            monkeypatch, "edit", "make it green", _DATA_URI, quality="low"
        ), "ImageClient.edit now accepts quality — revisit the adapter's Base-branch guard"
