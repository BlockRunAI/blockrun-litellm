"""
blockrun-litellm — LiteLLM adapter for BlockRun.

Two ways to pay, chosen by which credential is present:

* **API key** — ``BLOCKRUN_API_KEY=brk_live_...`` (issued at
  https://user.blockrun.ai, topped up by card). Calls go to
  ``api.blockrun.ai`` and are billed against prepaid credit. No wallet, no
  chain, no USDC.
* **x402 wallet** — ``SOLANA_WALLET_KEY`` (default chain) or
  ``BLOCKRUN_WALLET_KEY`` (Base). Each call is signed locally by the
  ``blockrun-llm`` SDK and settles on chain; your private key never leaves the
  host. No account needed.

Two integration modes, both of which work on either rail:

1. **Custom provider** (in-process):

    >>> import litellm
    >>> from blockrun_litellm import register
    >>> register()  # adds "blockrun/" provider to LiteLLM
    >>> resp = litellm.completion(
    ...     model="blockrun/openai/gpt-5.5",
    ...     messages=[{"role": "user", "content": "Hello"}],
    ... )

2. **Local OpenAI-compatible proxy** (sidecar):

    $ blockrun-litellm-proxy --port 4001 --api-key brk_live_...
    # then point LiteLLM at http://localhost:4001/v1
"""

from blockrun_litellm.logger import enable_local_logging
from blockrun_litellm.provider import BlockRunLLM, register
from blockrun_litellm.catalog import (
    BLOCKRUN_MODEL_IDS,
    CATALOG_SNAPSHOT_DATE,
    is_known_model,
    model_ids,
)

__all__ = [
    "BLOCKRUN_MODEL_IDS",
    "CATALOG_SNAPSHOT_DATE",
    "BlockRunLLM",
    "enable_local_logging",
    "is_known_model",
    "model_ids",
    "register",
]
__version__ = "0.10.0"
