"""Bundled BlockRun model-catalog snapshot.

The gateway is the source of truth and accepts new model IDs without a package
upgrade. This module is deliberately a static, dependency-free snapshot for
callers that need to render an allowlist or model picker before starting a
proxy. Call ``GET https://blockrun.ai/api/v1/models`` for live metadata;
``CATALOG_SNAPSHOT_DATE`` tells a caller how old this copy is.
"""

from __future__ import annotations

from typing import Tuple


# Snapshot taken from https://blockrun.ai/api/v1/models on the date below, in
# the gateway's own catalog order. Refresh both together. Only ``provider/model``
# ids are kept: the catalog also lists ``openjev``, a judgment endpoint rather
# than a model you can pass to completion().
CATALOG_SNAPSHOT_DATE = "2026-10-02"

BLOCKRUN_MODEL_IDS: Tuple[str, ...] = (
    "openai/gpt-6-astra",
    "openai/gpt-6-sol",
    "openai/gpt-6-luna",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-luna-pro",
    "openai/gpt-5.5",
    "openai/gpt-5.5-pro",
    "openai/chat-latest",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4-mini",
    "openai/gpt-5-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.2-pro",
    "openai/gpt-5.3-codex",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o1",
    "openai/o3",
    "openai/o3-mini",
    "openai/o4-mini",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-5.5",
    "google/gemini-3.1-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.8-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    "moonshot/kimi-k3",
    "zai/glm-5.3",
    "zai/glm-5.3-flash",
    "zai/glm-5.2",
    "zai/glm-5.1",
    "zai/glm-5",
    "zai/glm-5-turbo",
    "xai/grok-4.3",
    "xai/grok-build-0.1",
    "xai/grok-4.7",
    "xai/grok-4.6",
    "xai/grok-4.5",
    "minimax/minimax-m2.7",
    "minimax/minimax-m3",
    "qwen/qwen3.7-max",
    "qwen/qwen3.7-plus",
    "qwen/qwen3.7-flash",
    "qwen/qwen3.8-flash",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-pro",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-lightning",
    "nvidia/llama-3.2-11b-vision",
    "nvidia/nemotron-3-ultra-550b",
    "cohere/north-mini-code",
    "poolside/laguna-xs-2.1",
    "openai/gpt-image-1",
    "openai/gpt-image-2",
    "openai/gpt-image-2.5-flare",
    "openai/gpt-image-2.5-sunburst",
    "google/nano-banana",
    "google/nano-banana-2",
    "google/nano-banana-pro",
    "xai/grok-imagine-image",
    "xai/grok-imagine-image-2.0",
    "xai/grok-imagine-image-pro",
    "bytedance/seedream-5-pro",
    "zai/cogview-4",
    "minimax/music-2.5+",
    "elevenlabs/flash-v2.5",
    "elevenlabs/turbo-v2.5",
    "elevenlabs/multilingual-v2",
    "elevenlabs/v3",
    "bytedance/seed-audio-1.0",
    "elevenlabs/sound-effects",
    "xai/grok-imagine-video",
    "xai/grok-imagine-video-1.5",
    "bytedance/seedance-1.5-pro",
    "bytedance/seedance-2.0-fast",
    "bytedance/seedance-2.0-mini",
    "bytedance/seedance-2.0",
    "bytedance/seedance-2.5",
    "azure/sora-2",
)


def model_ids() -> Tuple[str, ...]:
    """Return the bundled model IDs in the gateway's catalog order."""
    return BLOCKRUN_MODEL_IDS


def is_known_model(model: str) -> bool:
    """Return whether ``model`` is in the bundled snapshot.

    Both the custom-provider ``blockrun/<id>`` form and the gateway's direct
    ``<id>`` form are accepted. This helper never blocks forwarding of a newly
    released ID; it is only useful for local UI and configuration validation.
    """
    return model.removeprefix("blockrun/") in BLOCKRUN_MODEL_IDS
