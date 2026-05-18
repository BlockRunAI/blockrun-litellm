# blockrun-litellm

[![PyPI](https://img.shields.io/pypi/v/blockrun-litellm.svg)](https://pypi.org/project/blockrun-litellm/)
[![Python](https://img.shields.io/pypi/pyversions/blockrun-litellm.svg)](https://pypi.org/project/blockrun-litellm/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

LiteLLM adapter for [BlockRun](https://blockrun.ai) — call x402-paid AI models through [LiteLLM](https://github.com/BerriAI/litellm) with zero changes to your existing code. **Base and Solana chains supported.**

📚 **Full docs in [`docs/`](docs/)** — bilingual (English + 中文):
- [`CUSTOMER-ONBOARDING`](docs/CUSTOMER-ONBOARDING.md) / [`中文`](docs/CUSTOMER-ONBOARDING.zh.md) — 5-minute walkthrough, both modes
- [`PROXY-FULL-SETUP`](docs/PROXY-FULL-SETUP.md) / [`中文`](docs/PROXY-FULL-SETUP.zh.md) — full deploy with admin UI + Postgres + troubleshooting

> **TL;DR** — BlockRun's `/v1/chat/completions` is already OpenAI-compatible at the protocol level. The only thing that differs is *authentication*: BlockRun uses per-request x402 wallet signatures (non-custodial USDC micropayments on Base / Solana), not a Bearer API key. This package bridges that gap.

[中文文档见底部 / Chinese docs at the bottom](#中文文档)

---

## Two ways to integrate

| Mode | Best for | What it looks like |
|---|---|---|
| **1. Custom provider** (in-process) | Apps using the LiteLLM **Python library** | `litellm.completion(model="blockrun/openai/gpt-5.5", ...)` |
| **2. Local proxy** (sidecar) | Apps using the LiteLLM **Proxy Server** (or any OpenAI client) | `api_base="http://localhost:4001/v1"` |

Both modes share the same underlying wallet/signing flow (via the [`blockrun-llm`](https://github.com/BlockRunAI/blockrun-llm) SDK), so they behave identically. Pick whichever fits your deployment.

### Verified end-to-end against the live BlockRun gateway

Both modes have been validated against `https://blockrun.ai/api` using the free `nvidia/deepseek-v4-flash` model:

```
$ python -c "
> import litellm
> from blockrun_litellm import register; register()
> r = litellm.completion(
>     model='blockrun/nvidia/deepseek-v4-flash',
>     messages=[{'role':'user','content':'Reply with exactly: pong'}],
>     max_tokens=20, temperature=0.0)
> print(r.choices[0].message.content)"
pong

$ curl -sS http://127.0.0.1:4001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"nvidia/deepseek-v4-flash","messages":[{"role":"user","content":"Reply with exactly: proxy-ok"}]}'
{"id":"a710c144c68c42f7a319fb93e9b9b5a0","object":"chat.completion","model":"nvidia/deepseek-v4-flash",
 "choices":[{"index":0,"message":{"role":"assistant","content":"proxy-ok"},...}],"usage":{...}}
```

---

## Install

```bash
# Base chain only — minimal
pip install blockrun-litellm

# Base chain + local OpenAI-compatible proxy (FastAPI/uvicorn)
pip install 'blockrun-litellm[proxy]'

# Base + Solana (adds the x402 SVM toolchain)
pip install 'blockrun-litellm[proxy,solana]'
```

Requires Python ≥ 3.9.

## Chains supported

| Chain | Gateway URL | Wallet env var | Status |
|---|---|---|---|
| Base (USDC) | `https://blockrun.ai/api` *(default)* | `BLOCKRUN_WALLET_KEY` | sync + async, streaming |
| Solana (USDC) | `https://sol.blockrun.ai/api` | `SOLANA_WALLET_KEY` | sync + async, streaming on both (since 0.3.1) |

To route on Solana, pass `api_base="https://sol.blockrun.ai/api"` plus `api_key=<solana-key>` to `litellm.completion(...)` — the adapter detects the chain from the URL and uses the right SDK client.

---

## Configure your wallet (one-time)

The `blockrun-llm` SDK signs each request locally with an EVM (Base chain) private key. **The key never leaves your machine.** Three ways to provide it:

```bash
# Option A — environment variable (recommended for servers)
export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_CHAIN_PRIVATE_KEY

# Option B — auto-create + fund a new wallet (interactive, shows QR for funding)
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"

# Option C — pass per-call (Python lib mode), see examples below
```

> 💡 To validate without spending real USDC, use a free model like `nvidia/deepseek-v4-flash` — same code path, same wallet flow, $0 settlement.

---

## Mode 1 — Custom provider (Python library)

The shortest path if your app already calls `litellm.completion()` directly.

### 1a. Register once at startup

```python
import litellm
from blockrun_litellm import register

register()  # idempotent; adds "blockrun" to litellm.custom_provider_map
```

### 1b. Call with a `blockrun/` model prefix

```python
response = litellm.completion(
    model="blockrun/openai/gpt-5.5",        # blockrun/<provider>/<model>
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=128,
    temperature=0.7,
)

print(response.choices[0].message.content)
print(response.usage)  # prompt_tokens / completion_tokens / total_tokens
```

The `blockrun/` prefix is stripped before being sent to the BlockRun gateway, so `openai/gpt-5.5`, `anthropic/claude-opus-4-5`, `google/gemini-3-pro`, etc. all work — anything in BlockRun's catalog.

### 1c. Override the wallet per-call (optional)

```python
response = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[...],
    api_key="0xANOTHER_PRIVATE_KEY",          # passed to blockrun-llm as wallet
)
```

### 1d. Async

```python
import asyncio

async def main():
    response = await litellm.acompletion(
        model="blockrun/openai/gpt-5.5",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## Mode 2 — Local proxy (LiteLLM Proxy Server, langchain, raw curl, …)

If you're running the **LiteLLM Proxy Server** (`litellm --config config.yaml`), or any client that just speaks OpenAI HTTP, run our proxy as a sidecar.

### 2a. Start the proxy

```bash
export BLOCKRUN_WALLET_KEY=0xYOUR_KEY
blockrun-litellm-proxy --port 4001
# → uvicorn running at http://127.0.0.1:4001
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind interface. **Keep loopback** unless you set `BLOCKRUN_PROXY_TOKEN`. |
| `--port` | `4001` | Bind port |
| `--api-url` | `https://blockrun.ai/api` | Override BlockRun gateway endpoint |
| `--log-level` | `info` | `critical`/`error`/`warning`/`info`/`debug`/`trace` |

Optional shared-secret guard:

```bash
export BLOCKRUN_PROXY_TOKEN=$(openssl rand -hex 32)
# clients must now send:  Authorization: Bearer $BLOCKRUN_PROXY_TOKEN
```

### 2b. Point LiteLLM Proxy at it

Drop this into your `config.yaml`:

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: openai/openai/gpt-5.5   # first 'openai/' = LiteLLM provider; rest = BlockRun model id
      api_base: http://localhost:4001/v1
      api_key: "dummy"                # ignored if BLOCKRUN_PROXY_TOKEN is unset

  - model_name: claude-opus-4-7
    litellm_params:
      model: openai/anthropic/claude-opus-4-7
      api_base: http://localhost:4001/v1
      api_key: "dummy"

  - model_name: gemini-3.1-pro
    litellm_params:
      model: openai/google/gemini-3.1-pro
      api_base: http://localhost:4001/v1
      api_key: "dummy"

litellm_settings:
  drop_params: True   # silently drop OpenAI params BlockRun doesn't support
```

Run LiteLLM Proxy as usual:

```bash
litellm --config config.yaml --port 4000
```

Then call it like any OpenAI endpoint:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 2c. Or skip LiteLLM entirely

The proxy speaks OpenAI HTTP, so anything that takes an `api_base` works:

```python
# OpenAI Python SDK pointed straight at the BlockRun proxy
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://localhost:4001/v1")
resp = client.chat.completions.create(
    model="openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
```

```bash
# Plain curl
curl http://localhost:4001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-5.5", "messages": [{"role":"user","content":"Hi"}]}'
```

### 2d. Endpoints exposed

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions. `stream=True` returns `text/event-stream`; otherwise JSON. |
| `POST` | `/v1/images/generations` | OpenAI Image Generations. Accepts `prompt`, `model`, `size`, `n`. |
| `GET`  | `/v1/models` | BlockRun model catalog |
| `GET`  | `/healthz` | Liveness probe (no upstream call) |
| `GET`  | `/docs` | Auto-generated Swagger UI |

### 2e. Image generation

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://localhost:4001/v1")
resp = client.images.generate(
    model="google/nano-banana",
    prompt="a corgi astronaut on the moon",
    size="1024x1024",
    n=1,
)
print(resp.data[0].url)
```

```bash
curl http://localhost:4001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "google/nano-banana", "prompt": "a corgi astronaut", "size": "1024x1024"}'
```

Available image models: `google/nano-banana`, `google/nano-banana-pro`, `openai/dall-e-3`, `openai/gpt-image-1`, `openai/gpt-image-2`, `xai/grok-imagine-image`, `xai/grok-imagine-image-pro`, `zai/cogview-4`.

---

## Supported parameters

All of these are forwarded to BlockRun unchanged:

| OpenAI param | Supported | Notes |
|---|---|---|
| `model` | ✅ | Any BlockRun model id, e.g. `openai/gpt-5.5` |
| `messages` | ✅ | Full role/content/tool_calls schema |
| `max_tokens` | ✅ | Defaults to 1024 if omitted |
| `temperature` | ✅ | 0–2 |
| `top_p` | ✅ | |
| `tools` / `tool_choice` | ✅ | Function calling |
| `stream` | ✅ | OpenAI-style SSE (`text/event-stream`). Provider mode yields LiteLLM `GenericStreamingChunk` objects; proxy mode emits `data: <json>\n\n` events terminated by `data: [DONE]`. Free models stream directly; paid models stream after the in-band 402-sign-retry dance. |
| `frequency_penalty` / `presence_penalty` / `logprobs` / `n` | ⚠️ | Silently dropped — enable `litellm_settings.drop_params: True` to suppress LiteLLM warnings |

BlockRun-specific extras (also accepted):

| Param | Purpose |
|---|---|
| `search: True` | Enable xAI Live Search (for search-enabled models) |
| `search_parameters: {...}` | Full Live Search config |
| `fallback_models: ["..."]` | Auto-retry on transient upstream errors |

---

## Local request log (input/output tokens, latency, cost)

Opt-in JSONL logger captures every call — works on both Base and Solana, sync and async, streaming and non-streaming.

### Where the log lives

| Source | Path |
|---|---|
| Explicit arg to `enable_local_logging("...")` | (whatever you pass) |
| `BLOCKRUN_LITELLM_LOG` env var | (whatever it points to) |
| Otherwise | **`~/.blockrun/litellm_calls.jsonl`** |

### Each row contains

```
ts, iso, model, provider, messages, completion,
usage{prompt_tokens, completion_tokens, total_tokens},
latency_ms, stream, cost_usd, status, error_type, error_message, request_id
```

### Mode 1 — one line

```python
from blockrun_litellm import enable_local_logging
enable_local_logging()                       # default path
# or enable_local_logging("/var/log/calls.jsonl")
```

### Mode 2 — drop a bridge file next to `config.yaml`

```python
# custom_callbacks.py
from blockrun_litellm.logger import JSONLLogger
blockrun_logger = JSONLLogger()
```

```yaml
litellm_settings:
  callbacks: ["custom_callbacks.blockrun_logger"]
```

---

## Where everything is stored

| File / env var | What | Configurable? |
|---|---|---|
| `BLOCKRUN_WALLET_KEY` (env) | Base private key | yes |
| `SOLANA_WALLET_KEY` (env) | Solana private key | yes |
| `~/.blockrun/.session` | Auto-created Base wallet | — |
| `~/.blockrun/.solana-session` | Auto-created Solana wallet | — |
| `~/.blockrun/litellm_calls.jsonl` | LiteLLM request log | `BLOCKRUN_LITELLM_LOG` env or `enable_local_logging(path)` |
| `~/.blockrun/cost_log.jsonl` | USDC cost audit for paid calls (SDK) | — |
| `~/.blockrun/data/*.json` | Full request/response archive for paid calls (SDK) | — |
| `BLOCKRUN_PROXY_TOKEN` (env) | Optional shared-secret guard on sidecar | yes |

---

## Examples

The `examples/` directory has copy-paste-ready snippets:

- [`examples/python_lib.py`](examples/python_lib.py) — full LiteLLM Python library usage
- [`examples/litellm_config.yaml`](examples/litellm_config.yaml) — LiteLLM Proxy Server config
- [`examples/raw_openai_sdk.py`](examples/raw_openai_sdk.py) — pointing the OpenAI SDK at the proxy
- [`examples/custom_callbacks.py`](examples/custom_callbacks.py) — JSONL log bridge for Proxy mode

---

## How it works (under the hood)

```
┌─────────────────┐    OpenAI dict     ┌──────────────────────┐    POST /v1/chat/completions  ┌────────────────┐
│ Your app /      │ ─────────────────▶ │  blockrun-litellm    │ ────────────────────────────▶ │  blockrun.ai   │
│ LiteLLM /       │                    │  (provider OR proxy) │ ◀──── 402 + payment-required ─│  gateway       │
│ OpenAI SDK      │                    │  ↓                   │                               │                │
└─────────────────┘                    │  blockrun-llm SDK    │ ───── EIP-712 signed retry ──▶│                │
                                       │  (local signing)     │ ◀──── 200 + chat response ────│                │
                                       └──────────────────────┘                               └────────────────┘
                                                ▲
                                                │ private key (stays local, signs only)
                                       ┌──────────────────────┐
                                       │ BLOCKRUN_WALLET_KEY  │
                                       │   or ~/.blockrun/    │
                                       └──────────────────────┘
```

1. Caller sends an OpenAI Chat Completions dict.
2. `blockrun-litellm` whitelists the params and dispatches through `blockrun-llm`.
3. `blockrun-llm` posts to BlockRun, receives a 402 with payment requirements, signs an EIP-712 payment locally with your wallet, and retries.
4. BlockRun verifies the signature on-chain, settles the USDC micropayment, runs the inference, and returns the response.
5. `blockrun-litellm` returns the dumped pydantic model as a plain OpenAI dict (or `litellm.ModelResponse` in provider mode).

---

## FAQ

**Q: Does this support streaming?**
Yes, as of v0.2.0. Pass `stream=True` and the adapter routes through `blockrun-llm`'s `chat_completion_stream()` (SDK ≥ 0.20.0). The 402 → sign-locally → retry-with-PAYMENT-SIGNATURE dance happens before the first chunk; once the upstream switches to `text/event-stream`, chunks are forwarded straight through (provider mode → `litellm.GenericStreamingChunk`, proxy mode → OpenAI-style `data: <json>\n\n` SSE). Caveats inherited from the gateway: `search_parameters` and the Responses-API models (`codex`, `gpt-5.4-pro`) reject streaming server-side with 400.

**Q: Where does my private key live?**
On your machine only — `BLOCKRUN_WALLET_KEY` env var, or `~/.blockrun/.session` if you used `setup_agent_wallet()`. The proxy and provider both read from those sources via `blockrun-llm`. Only EIP-712 signatures are transmitted.

**Q: How do I switch between Base and Solana?**
Today this adapter wires to BlockRun's Base gateway (USDC on Base). Solana support tracks the `blockrun-llm` `SolanaLLMClient` and will be added in a follow-up release.

**Q: Can I run the proxy in Docker / k8s?**
Yes — it's a vanilla FastAPI app. Pass the wallet key via secret (env var), bind to `0.0.0.0` only inside a private network, and set `BLOCKRUN_PROXY_TOKEN` for an additional auth layer.

**Q: Is this affiliated with LiteLLM (BerriAI)?**
No — this is an independent adapter built by the BlockRun team. LiteLLM is a great project; we're just plugging into its custom-provider hooks.

---

## Development

```bash
git clone https://github.com/BlockRunAI/blockrun-litellm
cd blockrun-litellm
pip install -e '.[proxy,dev]'
pytest
```

---

## License

MIT. See [LICENSE](LICENSE).

---

# 中文文档

[BlockRun](https://blockrun.ai) 的 [LiteLLM](https://github.com/BerriAI/litellm) 适配层 —— 用 LiteLLM 调用 BlockRun 上的 AI 模型，**完全零改动**。

> **一句话：** BlockRun 的 `/v1/chat/completions` 协议层就是 OpenAI 兼容的，唯一区别是认证方式 —— BlockRun 用 x402 钱包签名（按次 USDC 微支付，非托管），不是 Bearer API Key。这个包就是把这层差异填平。

## 两种对接方式

| 模式 | 适用 | 写法 |
|---|---|---|
| **1. 自定义 Provider**（进程内） | 用 LiteLLM **Python 库**的应用 | `litellm.completion(model="blockrun/openai/gpt-5.5", ...)` |
| **2. 本地代理**（sidecar） | 用 LiteLLM **Proxy Server** 的、或任何 OpenAI 客户端 | `api_base="http://localhost:4001/v1"` |

底层都走 [`blockrun-llm`](https://github.com/BlockRunAI/blockrun-llm) SDK 做签名和 x402 支付，两种模式行为一致。按你的部署方式选一种就行。

## 快速上手

### 安装

```bash
# 只装自定义 provider
pip install blockrun-litellm

# 同时装本地代理（带 FastAPI/uvicorn）
pip install 'blockrun-litellm[proxy]'
```

### 配钱包（一次性）

```bash
# 方式 A — 环境变量（服务端推荐）
export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_CHAIN_PRIVATE_KEY

# 方式 B — 自动创建并扫码充值（交互式）
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"
```

私钥**只在本地用于 EIP-712 签名**，永远不会离开你的机器。

> 💡 想零成本试一遍？用免费模型 `nvidia/deepseek-v4-flash` —— 代码完全一样，钱包流程一样，结算 $0。

### 模式 1：自定义 Provider

```python
import litellm
from blockrun_litellm import register

register()  # 启动时调一次即可

response = litellm.completion(
    model="blockrun/openai/gpt-5.5",   # blockrun/<provider>/<model>
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

异步版本：`await litellm.acompletion(...)` 同理。

### 模式 2：本地代理

```bash
# 1) 启动 sidecar
export BLOCKRUN_WALLET_KEY=0xYOUR_KEY
blockrun-litellm-proxy --port 4001

# 2) LiteLLM Proxy 配置 (config.yaml)
```

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: openai/openai/gpt-5.5
      api_base: http://localhost:4001/v1
      api_key: "dummy"

litellm_settings:
  drop_params: True
```

或者直接拿任何 OpenAI 客户端用：

```python
from openai import OpenAI
client = OpenAI(api_key="dummy", base_url="http://localhost:4001/v1")
resp = client.chat.completions.create(
    model="openai/gpt-5.5",
    messages=[{"role": "user", "content": "你好"}],
)
```

## 支持的参数

| OpenAI 参数 | 支持 | 备注 |
|---|---|---|
| `model` / `messages` / `max_tokens` / `temperature` / `top_p` | ✅ | |
| `tools` / `tool_choice` | ✅ | 函数调用 |
| `stream` | ✅ | OpenAI 标准 SSE（`text/event-stream`）。Provider 模式 yield LiteLLM `GenericStreamingChunk`；Proxy 模式发 `data: <json>\n\n` 事件并以 `data: [DONE]` 结尾。免费模型直接开流；付费模型走带内 402→签名→重试再开流。 |
| `frequency_penalty` / `presence_penalty` / `logprobs` / `n` | ⚠️ | 静默丢弃 —— 建议 LiteLLM 配 `drop_params: True` 抑制告警 |

BlockRun 额外参数：

| 参数 | 作用 |
|---|---|
| `search: True` | 启用 xAI Live Search（搜索类模型） |
| `search_parameters: {...}` | 完整 Live Search 配置 |
| `fallback_models: ["..."]` | 上游抖动自动重试列表 |

## 常见问题

**Q：支持流式吗？**
v0.2.0 起完全支持。`stream=True` 时适配层走 `blockrun-llm` 的 `chat_completion_stream()`（SDK ≥ 0.20.0），402 → 本地签名 → 带 PAYMENT-SIGNATURE 重试这条链在第一个 chunk 之前完成；上游切到 `text/event-stream` 后 chunks 直接透传（Provider 模式 → `litellm.GenericStreamingChunk`，Proxy 模式 → OpenAI 标准 `data: <json>\n\n`）。后端继承的限制：`search_parameters` 和 Responses-API 模型（`codex`、`gpt-5.4-pro`）在服务端就拒绝流式（400）。

**Q：私钥放哪？**
只在本地 —— `BLOCKRUN_WALLET_KEY` 环境变量，或 `setup_agent_wallet()` 创建的 `~/.blockrun/.session`。Provider 和 Proxy 都通过 `blockrun-llm` 读取。链上只看到签名，看不到私钥。

**Q：Docker / k8s 部署？**
代理是普通的 FastAPI 应用。密钥用 secret 注入，对外只暴露内网，可选 `BLOCKRUN_PROXY_TOKEN` 加一层 Bearer 鉴权。

**Q：和 BerriAI 是什么关系？**
没关系。这是 BlockRun 团队独立维护的适配层，挂在 LiteLLM 的 custom provider 接口上。

## 开发

```bash
git clone https://github.com/BlockRunAI/blockrun-litellm
cd blockrun-litellm
pip install -e '.[proxy,dev]'
pytest
```

## License

MIT
