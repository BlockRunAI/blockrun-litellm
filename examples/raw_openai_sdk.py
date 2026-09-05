"""
Mode 2 example — point the official OpenAI SDK at the blockrun-litellm proxy.

Prereqs:
    pip install 'blockrun-litellm[proxy]' openai

Start the sidecar in another terminal, with ONE credential:
    blockrun-litellm-proxy --port 4001 --api-key brk_live_...   # account credit
    # or:
    export SOLANA_WALLET_KEY=YOUR_SOLANA_KEY                    # x402, Solana (default)
    blockrun-litellm-proxy --port 4001

Then:
    python examples/raw_openai_sdk.py
"""

from openai import OpenAI


def main() -> None:
    # api_key is required by the OpenAI SDK constructor but is ignored by our
    # proxy unless BLOCKRUN_PROXY_TOKEN is set on the sidecar.
    client = OpenAI(api_key="dummy", base_url="http://localhost:4001/v1")

    # List models — passes through to BlockRun's /v1/models catalog.
    models = client.models.list()
    print(f"[models] {len(list(models))} BlockRun models available")

    # Chat completion — uses the BlockRun model id directly (no "blockrun/" prefix here).
    resp = client.chat.completions.create(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "What is 17 * 23?"}],
        max_tokens=32,
    )
    print("[answer]", resp.choices[0].message.content)


if __name__ == "__main__":
    main()
