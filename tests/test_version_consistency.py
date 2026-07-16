"""The package version must be declared identically in both places.

Releases bump it in two files — ``pyproject.toml`` (what PyPI ships) and
``blockrun_litellm/__init__.py`` (what ``blockrun_litellm.__version__``
reports). They drift: 0.4.2 went stale until 0.5.0 noticed, and 0.7.1 bumped
pyproject while leaving ``__version__`` at "0.7.0" — caught here only because
0.7.1 hadn't been published yet.

blockrun-llm has carried this guard since its own 1.4.6 drift. This is the
same test for the sidecar, which had none.

Parsed with a regex rather than tomllib so it runs on Python 3.9 (no stdlib
TOML parser before 3.11) without adding a tomli dependency.
"""

import re
from pathlib import Path

import blockrun_litellm

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    assert match, "no version declared in pyproject.toml"
    return match.group(1)


def test_version_matches_pyproject():
    assert blockrun_litellm.__version__ == _pyproject_version(), (
        f"version drift: __init__.py={blockrun_litellm.__version__!r} "
        f"!= pyproject.toml={_pyproject_version()!r} — bump BOTH on release"
    )
