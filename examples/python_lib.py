"""
Mode 1 example — LiteLLM Python library with the blockrun custom provider.

Prereqs:
    pip install blockrun-litellm
    export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_CHAIN_PRIVATE_KEY

Run:
    python examples/python_lib.py
"""

import asyncio

import litellm

from blockrun_litellm import register


def sync_example() -> None:
    response = litellm.completion(
        model="blockrun/openai/gpt-5.5",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name three primary colors."},
        ],
        max_tokens=64,
        temperature=0.2,
    )
    print("[sync]", response.choices[0].message.content)
    print("[usage]", response.usage)


async def async_example() -> None:
    response = await litellm.acompletion(
        model="blockrun/anthropic/claude-opus-4-5",
        messages=[{"role": "user", "content": "Reverse the word 'banana'."}],
        max_tokens=32,
    )
    print("[async]", response.choices[0].message.content)


def tool_calling_example() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    response = litellm.completion(
        model="blockrun/openai/gpt-5.5",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
        tool_choice="auto",
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"[tool_call] {tc.function.name}({tc.function.arguments})")
    else:
        print("[no_tool]", msg.content)


if __name__ == "__main__":
    register()  # idempotent
    sync_example()
    asyncio.run(async_example())
    tool_calling_example()
