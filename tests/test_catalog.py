import re

from blockrun_litellm.catalog import (
    BLOCKRUN_MODEL_IDS,
    CATALOG_SNAPSHOT_DATE,
    is_known_model,
    model_ids,
)


def test_model_catalog_matches_live_snapshot() -> None:
    assert len(BLOCKRUN_MODEL_IDS) == 109
    assert len(set(BLOCKRUN_MODEL_IDS)) == len(BLOCKRUN_MODEL_IDS)
    assert all(mid.count("/") == 1 and mid.strip() == mid for mid in BLOCKRUN_MODEL_IDS)
    assert "anthropic/claude-opus-5" in BLOCKRUN_MODEL_IDS
    assert "bytedance/seedream-5-pro" in BLOCKRUN_MODEL_IDS
    assert "azure/sora-2" in BLOCKRUN_MODEL_IDS


def test_catalog_carries_the_2026_10_chat_models() -> None:
    for mid in (
        "openai/gpt-6-astra",
        "openai/gpt-6-sol",
        "openai/gpt-6-luna",
        "openai/gpt-5.1",
        "anthropic/claude-fable-5.1",
        "anthropic/claude-opus-5.5",
        "anthropic/claude-sonnet-5.5",
        "xai/grok-4.6",
        "xai/grok-4.7",
    ):
        assert is_known_model(mid), mid
        assert is_known_model(f"blockrun/{mid}"), mid


def test_catalog_drops_ids_the_gateway_no_longer_lists() -> None:
    # Retired (tencent/hy3) or unavailable (zai/glm-5-code) on the live gateway.
    assert not is_known_model("tencent/hy3")
    assert not is_known_model("zai/glm-5-code")
    # Not a provider/model id, so it can never be a completion() target.
    assert "openjev" not in BLOCKRUN_MODEL_IDS


def test_catalog_helpers_accept_custom_provider_prefix() -> None:
    assert model_ids() == BLOCKRUN_MODEL_IDS
    assert is_known_model("anthropic/claude-opus-5")
    assert is_known_model("blockrun/anthropic/claude-opus-5")
    assert not is_known_model("anthropic/not-a-real-model")


def test_snapshot_date_is_iso_dated() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", CATALOG_SNAPSHOT_DATE)
