"""
Mode 2 example — point the official OpenAI SDK at the blockrun-litellm proxy.

Prereqs:
    pip install 'blockrun-litellm[proxy]' openai
    export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_CHAIN_PRIVATE_KEY

Start the sidecar in another terminal:
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
