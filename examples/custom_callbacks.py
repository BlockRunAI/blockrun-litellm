"""
Drop this file next to your LiteLLM Proxy ``config.yaml``, then reference
it from ``litellm_settings.callbacks`` like so::

    litellm_settings:
      callbacks: ["custom_callbacks.blockrun_logger"]

LiteLLM Proxy loads callbacks by filename-relative-to-config.yaml; it does
NOT import names directly from installed PyPI packages. This file is the
one-line bridge that imports our :class:`JSONLLogger` and exposes it under
a name LiteLLM Proxy can find.

Configure the destination via env var (read at process start):

    export BLOCKRUN_LITELLM_LOG=/var/log/blockrun-litellm-calls.jsonl
    litellm --config config.yaml
"""

from blockrun_litellm.logger import JSONLLogger

# The name on the right (``blockrun_logger``) is what you reference in
# config.yaml as ``custom_callbacks.blockrun_logger``.
blockrun_logger = JSONLLogger()
