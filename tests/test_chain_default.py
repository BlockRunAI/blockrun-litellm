"""Which chain an unconfigured host uses.

0.10.0 flipped the wallet-rail default from Base to Solana. The flip is only
half the behaviour: a host that has only ever held a Base wallet keeps using
Base, because upgrading a package must not brick a running deployment whose key
the Solana signer cannot even parse. Both halves are pinned here — the default
and the escape hatch — along with the precedence between them.

The conftest points the session-file probe at a directory that does not exist,
so "no wallet" means no wallet regardless of what is in the developer's home.
"""

from __future__ import annotations

import pytest

from blockrun_litellm import _adapter


@pytest.fixture(autouse=True)
def _clear_cache():
    # _default_wallet_api_url memoizes on the wallet env vars; each case here
    # changes them and must not see the previous case's answer.
    _adapter._reset_chain_cache_for_tests()
    yield
    _adapter._reset_chain_cache_for_tests()


def test_unconfigured_host_defaults_to_solana():
    assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL


def test_solana_wallet_keeps_the_solana_default(monkeypatch):
    monkeypatch.setenv("SOLANA_WALLET_KEY", "base58-looking-key")
    assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL


@pytest.mark.parametrize("env", ["BLOCKRUN_WALLET_KEY", "BASE_CHAIN_WALLET_KEY"])
def test_base_only_host_stays_on_base(monkeypatch, caplog, env):
    """The compatibility half of the flip.

    A deployment holding only a Base key would start failing inside the SVM
    signer the moment it upgraded. It keeps working — and is told, once, how to
    make the choice explicit.
    """
    monkeypatch.setenv(env, "0xdeadbeef")
    with caplog.at_level("WARNING"):
        assert _adapter.resolve_api_url() == _adapter.BASE_API_URL
    assert "BLOCKRUN_CHAIN=base" in caplog.text


def test_base_wallet_does_not_win_when_a_solana_wallet_exists(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_WALLET_KEY", "0xdeadbeef")
    monkeypatch.setenv("SOLANA_WALLET_KEY", "base58-looking-key")
    assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL


@pytest.mark.parametrize(
    "value,expected",
    [
        ("solana", _adapter.SOLANA_API_URL),
        ("sol", _adapter.SOLANA_API_URL),
        ("svm", _adapter.SOLANA_API_URL),
        ("base", _adapter.BASE_API_URL),
        ("evm", _adapter.BASE_API_URL),
        ("  BASE  ", _adapter.BASE_API_URL),
    ],
)
def test_blockrun_chain_selects_the_gateway(monkeypatch, value, expected):
    monkeypatch.setenv("BLOCKRUN_CHAIN", value)
    assert _adapter.resolve_api_url() == expected


def test_explicit_chain_beats_the_wallet_probe(monkeypatch):
    """``BLOCKRUN_CHAIN=solana`` on a Base-only host must NOT silently serve Base.

    The auto-detection exists to protect callers who never made a choice. Once
    someone has made one, honouring it — and letting the call fail with "no
    Solana wallet" — is the only answer that is not a lie about which chain
    just moved money.
    """
    monkeypatch.setenv("BLOCKRUN_WALLET_KEY", "0xdeadbeef")
    monkeypatch.setenv("BLOCKRUN_CHAIN", "solana")
    assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL


def test_unknown_chain_warns_and_falls_back_to_detection(monkeypatch, caplog):
    monkeypatch.setenv("BLOCKRUN_CHAIN", "ethereum")
    with caplog.at_level("WARNING"):
        assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL
    assert "not a known chain" in caplog.text


# ---------------------------------------------------------------------------
# The chain the CLI recorded, and the chain a key's own shape implies
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_files(monkeypatch, tmp_path):
    """Point the CLI chain files at a writable directory the test controls."""
    files = (tmp_path / "payment-chain", tmp_path / ".chain")
    monkeypatch.setattr(_adapter, "_CHAIN_FILES", files)
    return files


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "written,expected",
    [("base", _adapter.BASE_API_URL), ("solana", _adapter.SOLANA_API_URL), (" BASE\n", _adapter.BASE_API_URL)],
)
def test_a_chain_the_cli_recorded_is_honoured(chain_files, index, written, expected):
    """Someone who ran the setup flow has already answered this question.

    Ignoring `~/.blockrun/.chain` was a real bug in the first cut of 0.10.0:
    a user who picked Base interactively would have been silently moved to
    Solana by the new default, which is precisely the breakage the default's
    compatibility branch exists to prevent.
    """
    chain_files[index].write_text(written)
    assert _adapter.resolve_api_url() == expected


def test_the_current_chain_file_beats_the_legacy_one(chain_files):
    """Not hypothetical: both files exist on a real machine and disagree there
    (`payment-chain` solana, the older `.chain` base). One of them has to win,
    and it is the current name."""
    chain_files[0].write_text("solana")
    chain_files[1].write_text("base")
    assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL


def test_the_legacy_file_still_answers_on_its_own(chain_files):
    chain_files[1].write_text("base")
    assert _adapter.resolve_api_url() == _adapter.BASE_API_URL


def test_an_unreadable_chain_file_does_not_break_resolution(chain_files, caplog):
    chain_files[0].write_text("ethereum")
    with caplog.at_level("WARNING"):
        assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL
    assert "not a known chain" in caplog.text


def test_env_beats_the_recorded_chain(chain_files, monkeypatch):
    chain_files[0].write_text("base")
    monkeypatch.setenv("BLOCKRUN_CHAIN", "solana")
    assert _adapter.resolve_api_url() == _adapter.SOLANA_API_URL


@pytest.mark.parametrize(
    "key,expected",
    [
        ("0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", "base"),
        ("59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", "base"),
        ("5JGiBQ8m1nP1cWQ1kR5nJTaHqZ1YvWc8Vc1sQ3sQ3sQ3sQ3sQ3sQ", "solana"),
    ],
)
def test_an_explicit_key_picks_its_own_chain(chain_files, key, expected):
    """The key on the call is the wallet that will pay.

    Routing it to the other chain's signer cannot work — a hex key is not
    base58 — so the key's own format outranks anything recorded on disk.
    """
    chain_files[0].write_text("solana" if expected == "base" else "base")
    url = _adapter.resolve_api_url(private_key=key)
    assert url == (_adapter.BASE_API_URL if expected == "base" else _adapter.SOLANA_API_URL)


def test_api_url_env_beats_chain(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_CHAIN", "solana")
    monkeypatch.setenv("BLOCKRUN_API_URL", "https://blockrun.ai/api")
    assert _adapter.resolve_api_url() == "https://blockrun.ai/api"
    assert _adapter._is_solana_url(None) is False


def test_explicit_argument_beats_everything(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_CHAIN", "solana")
    monkeypatch.setenv("BLOCKRUN_API_URL", "https://sol.blockrun.ai/api")
    assert _adapter.resolve_api_url("https://staging.example/api") == "https://staging.example/api"
