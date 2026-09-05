# blockrun-litellm

[![PyPI](https://img.shields.io/pypi/v/blockrun-litellm.svg)](https://pypi.org/project/blockrun-litellm/)
[![Python](https://img.shields.io/pypi/pyversions/blockrun-litellm.svg)](https://pypi.org/project/blockrun-litellm/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

LiteLLM adapter for [BlockRun](https://blockrun.ai) — call 90+ AI models through [LiteLLM](https://github.com/BerriAI/litellm) with zero changes to your existing code. Pay with a **BlockRun API key** (card top-up, no wallet) or with an **x402 USDC wallet** on **Solana or Base**.

📚 **Full docs in [`docs/`](docs/)** — bilingual (English + 中文):
- [`CUSTOMER-ONBOARDING`](docs/CUSTOMER-ONBOARDING.md) / [`中文`](docs/CUSTOMER-ONBOARDING.zh.md) — 5-minute walkthrough, both modes
- [`PROXY-FULL-SETUP`](docs/PROXY-FULL-SETUP.md) / [`中文`](docs/PROXY-FULL-SETUP.zh.md) — full deploy with admin UI + Postgres + troubleshooting

🌐 **Hosted docs:** [**blockrun.ai/docs**](https://blockrun.ai/docs)
- [Chat Completions API](https://blockrun.ai/docs/api-reference/chat-completions)
- [Models & pricing](https://blockrun.ai/docs/api-reference/models)

> **TL;DR** — BlockRun's `/v1/chat/completions` is already OpenAI-compatible at the protocol level. What differs is *billing*. Two ways to settle: an ordinary API key billed against prepaid credit, or per-request x402 wallet signatures (non-custodial USDC on Solana / Base). This package handles both, and everything above the credential is identical.

[中文文档见底部 / Chinese docs at the bottom](#中文文档)

---

## Get an API key (30 seconds)

1. Sign in at **[user.blockrun.ai](https://user.blockrun.ai)** with Google.
2. **Billing → Add credit.** Card payment, $5 minimum. The processing fee (5.5% + $0.30) is charged at purchase, so every model then bills at the published list price — no per-call minimum, no per-call fee, no markup.
3. **API keys → Create key.** You get a `brk_live_…` key, shown once.

```bash
export BLOCKRUN_API_KEY=brk_live_...
```

That is the whole setup. No wallet, no chain, no USDC, no gas.

Prefer to pay from a wallet you control? Skip to [**Pay with an x402 wallet**](#pay-with-an-x402-wallet-solana--base) — it needs no account at all.

---

## Two ways to pay

|  | **API key** | **x402 wallet** |
|---|---|---|
| Set up | Sign in at [user.blockrun.ai](https://user.blockrun.ai), top up by card | Fund a wallet with USDC |
| Credential | `BLOCKRUN_API_KEY=brk_live_…` | `SOLANA_WALLET_KEY` / `BLOCKRUN_WALLET_KEY` |
| Endpoint | `https://api.blockrun.ai` | `https://sol.blockrun.ai/api` (default) or `https://blockrun.ai/api` |
| Billing | Prepaid credit, debited at list price | Per-call USDC settled on chain |
| Account needed | Yes | **No** |
| Chain | None — payment never touches a chain | Solana or Base |
| Per-call cost reported | No — see the ledger at [user.blockrun.ai](https://user.blockrun.ai) | **Yes** — the exact settled charge, per call |
| Where spend shows up | Dashboard → Activity | The chain, plus `x-blockrun-settlement` |
| Native Gemini (`/v1beta`) | Not available | Available |
| Extras to install | none | `[solana]` for the Solana signer |

Everything else is the same on both: the same model catalogue, the same OpenAI/Anthropic wire formats, the same streaming, the same native fingerprint passthrough.

**Precedence.** A key wins whenever one is present. `BLOCKRUN_API_KEY` (or `--api-key`, or `api_key="brk_live_…"` on a call) selects the API-key rail; with no key the adapter falls back to the wallet rail. Wallet keys are never confused with account keys — only a `brk_` prefix selects the account rail, and no private-key format starts with one.

---

## Two ways to integrate

| Mode | Best for | What it looks like |
|---|---|---|
| **1. Custom provider** (in-process) | Apps using the LiteLLM **Python library** | `litellm.completion(model="blockrun/openai/gpt-5.5", ...)` |
| **2. Local proxy** (sidecar) | Apps using the LiteLLM **Proxy Server** (or any OpenAI client) | `api_base="http://localhost:4001/v1"` |

Both modes work on both rails and behave identically. Pick whichever fits your deployment.

---

## Install

```bash
# API key, or a Solana/Base wallet used from the Python library
pip install blockrun-litellm

# ...plus the local OpenAI-compatible proxy (FastAPI/uvicorn)
pip install 'blockrun-litellm[proxy]'

# ...plus the x402 SVM signer, needed ONLY to pay from a Solana wallet
pip install 'blockrun-litellm[proxy,solana]'
```

Requires Python ≥ 3.9. On the API-key rail the `solana` extra is unnecessary — there is no signing to do.

---

## Quick start

### With an API key

```python
import litellm
from blockrun_litellm import register

register()  # idempotent; adds "blockrun" to litellm.custom_provider_map

# BLOCKRUN_API_KEY is read from the environment; or pass api_key= per call.
r = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
print(r.choices[0].message.content)
```

Or as a sidecar for anything that speaks OpenAI HTTP:

```bash
blockrun-litellm-proxy --port 4001 --api-key brk_live_...
curl -s http://127.0.0.1:4001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-5.5","messages":[{"role":"user","content":"hi"}]}'
```

If all you want is chat, you do not need this package at all on the API-key rail — `https://api.blockrun.ai/v1` is OpenAI-compatible, so any OpenAI SDK works by changing `base_url`. The package earns its place when you want LiteLLM routing/fallbacks, the local audit log, or the media and Anthropic surfaces under one proxy.

---

## Pay with an x402 wallet (Solana / Base)

No account, no sign-up: fund a wallet with USDC and every request settles itself. This is the agent-native path — an autonomous agent can hold a wallet, but it cannot fill in a card form.

### Chains supported

| Chain | Gateway URL | Wallet env var | Notes |
|---|---|---|---|
| **Solana (USDC)** — *default* | `https://sol.blockrun.ai/api` | `SOLANA_WALLET_KEY` | Sub-second settlement, lowest fee. Needs the `[solana]` extra. Sync + async, streaming. |
| Base (USDC) | `https://blockrun.ai/api` | `BLOCKRUN_WALLET_KEY` | Sync + async, streaming. |

**Solana is the default chain as of 0.10.0.** Previously an unconfigured host used Base. Pick a chain explicitly with `BLOCKRUN_CHAIN`:

```bash
export BLOCKRUN_CHAIN=solana   # default
export BLOCKRUN_CHAIN=base
```

or point at a gateway directly with `BLOCKRUN_API_URL` / `--api-url` / `api_base=` (which always wins over `BLOCKRUN_CHAIN`).

> **Upgrading from ≤ 0.9.x on Base?** Nothing breaks. When no chain is configured and the host holds *only* a Base wallet, the adapter keeps using Base and logs a one-line warning. Set `BLOCKRUN_CHAIN=base` to make the choice explicit and silence it.

### Configure your wallet (one-time)

The `blockrun-llm` SDK signs each request locally. **The key never leaves your machine** — only signatures travel.

```bash
# Option A — environment variable (recommended for servers)
export SOLANA_WALLET_KEY=YOUR_SOLANA_PRIVATE_KEY      # Solana (default chain)
export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_PRIVATE_KEY    # Base

# Option B — auto-create + fund a new wallet (interactive, shows QR for funding)
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"

# Option C — pass per-call (Python lib mode), see examples below
```

> 💡 To validate without spending real USDC, use a free model like `nvidia/deepseek-v4-flash` — same code path, same wallet flow, $0 settlement.

---

## What each rail can serve

Every surface below works on both rails except where noted.

| Surface | API key | Wallet |
|---|---|---|
| `POST /v1/chat/completions` (+ streaming) | ✅ | ✅ |
| `POST /v1/messages` — native Anthropic | ✅ | ✅ |
| `POST /v1/responses` — OpenAI Responses | ✅ | ✅ |
| `POST /v1/images/generations`, `/v1/images/edits` | ✅ | ✅ |
| `POST /v1/videos`, `/v1/videos/generations` (+ poll, download) | ✅ | ✅ |
| `POST /v1/audio/speech`, `/v1/audio/generations`, `/v1/audio/sound-effects` | ✅ | ✅ |
| `GET /v1/models` | ✅ | ✅ |
| `POST /v1beta/models/{model}:generateContent` — **native Gemini** | ❌ 501 | ✅ |

Native Gemini is the one gap: `api.blockrun.ai` does not publish `/v1beta`. The proxy answers 501 with that explanation rather than a bare 404. Gemini models themselves are reachable on both rails through `/v1/chat/completions` (`model="google/gemini-3-pro"`); only Google's own protocol needs a wallet.

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

The `blockrun/` prefix is stripped before being sent to the BlockRun gateway, so `openai/gpt-5.6-terra`, `anthropic/claude-opus-5`, `google/gemini-3.1-pro`, etc. all work — anything in BlockRun's catalog.

For local allowlists and model pickers, the package includes the current 82-model
catalog snapshot (chat, image, video, music, speech, and sound effects):

```python
from blockrun_litellm import model_ids, is_known_model

assert "anthropic/claude-opus-5" in model_ids()
assert is_known_model("blockrun/anthropic/claude-opus-5")
```

The gateway is authoritative and accepts newly released IDs before a package
update; query `https://blockrun.ai/api/v1/models` whenever you need live metadata.

### 1c. Override the credential per-call (optional)

`api_key` carries either credential — the `brk_` prefix decides which rail serves the call, so one parameter covers both and existing wallet code is untouched:

```python
# Account rail — billed to prepaid credit
litellm.completion(model="blockrun/openai/gpt-5.5", messages=[...],
                   api_key="brk_live_...")

# Wallet rail — x402, signed locally
litellm.completion(model="blockrun/openai/gpt-5.5", messages=[...],
                   api_key="0xANOTHER_PRIVATE_KEY")

# Wallet rail, specific chain
litellm.completion(model="blockrun/openai/gpt-5.5", messages=[...],
                   api_base="https://blockrun.ai/api", api_key="0xBASE_KEY")
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
# API-key rail
blockrun-litellm-proxy --port 4001 --api-key brk_live_...

# x402 wallet rail (Solana by default)
export SOLANA_WALLET_KEY=YOUR_SOLANA_PRIVATE_KEY
blockrun-litellm-proxy --port 4001
# → uvicorn running at http://127.0.0.1:4001
```

The sidecar fails fast at startup if it has neither credential, and tells you which rail it picked.

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind interface. **Keep loopback** unless you set `BLOCKRUN_PROXY_TOKEN`. |
| `--port` | `4001` | Bind port |
| `--api-key` | *(unset)* | BlockRun API key (`brk_live_…`). Selects the account rail; no chain involved. Env: `BLOCKRUN_API_KEY` |
| `--chain` | `solana` | Wallet-rail chain: `solana` or `base`. Ignored with `--api-key`. Env: `BLOCKRUN_CHAIN` |
| `--api-url` | *(the `--chain` gateway)* | Override the gateway endpoint outright |
| `--log-level` | `info` | `critical`/`error`/`warning`/`info`/`debug`/`trace` |

Environment variables (no CLI flag):

| Env var | Default | Purpose |
|---|---|---|
| `BLOCKRUN_API_KEY` | *(unset)* | Account key. When set, every route serves from prepaid credit. |
| `BLOCKRUN_API_BASE_URL` | `https://api.blockrun.ai` | Account API endpoint (staging overrides). |
| `BLOCKRUN_CHAIN` | `solana` | Wallet-rail chain. |
| `BLOCKRUN_MAX_CONCURRENT` | `100` | Max in-flight requests. Excess requests queue inside the sidecar. See table below for tuning guidance. |
| `BLOCKRUN_PROXY_TOKEN` | *(unset)* | Optional Bearer token guard on all sidecar endpoints. Never forwarded upstream. |

> **The credential stays in the sidecar.** Clients on the same host send `BLOCKRUN_PROXY_TOKEN` (if you set one) and never see your API key or wallet key — the sidecar strips the client's `Authorization` header before forwarding and substitutes its own.

#### High-concurrency tuning

Streaming requests release their semaphore slot as soon as the first token arrives from upstream (the x402 probe + sign is the only serialised part). For non-streaming requests the slot is held until the full response returns.

| Deployment | `BLOCKRUN_MAX_CONCURRENT` | `uvicorn --workers` | Effective max concurrency |
|---|---|---|---|
| Dev / single user | 20 | 1 | 20 |
| Small team (10–20 concurrent) | 50 | 1 | 50 |
| Production (100 concurrent) | 100 *(default)* | 1 | 100 |
| High-load (500+ concurrent) | 200 | 4 | 800 |

Multi-worker launch example:
```bash
BLOCKRUN_MAX_CONCURRENT=200 uvicorn blockrun_litellm.proxy:app --workers 4 --host 0.0.0.0 --port 4001
```

> **Note:** Each worker has its own semaphore. With `--workers 4` and `BLOCKRUN_MAX_CONCURRENT=200`, up to 800 requests can be in-flight simultaneously. The real ceiling is the upstream provider's RPM/TPM — see the [Enterprise SLA guide](docs/ENTERPRISE-SLA.zh.md) for per-provider limits.

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

  - model_name: claude-fable-5
    litellm_params:
      model: openai/anthropic/claude-fable-5
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

### 2b-ii. Image / video models through LiteLLM — routing **and** billing

Two things trip people up when they add BlockRun's media models
(`xai/grok-imagine-image`, `xai/grok-imagine-image-pro`,
`xai/grok-imagine-video`, …) to a LiteLLM proxy:

1. **LiteLLM logs $0 spend for them.** LiteLLM prices calls from its bundled
   price map (`model_prices_and_context_window.json`), which has **no**
   BlockRun-routed media models — the cost lookup fails and the request is
   recorded at $0. Fix: declare the price in `litellm_params` (LiteLLM's
   [custom pricing](https://docs.litellm.ai/docs/proxy/custom_pricing)).
   LiteLLM bills images as `input_cost_per_pixel × width × height × n`, so a
   flat per-image price divides by 1024×1024 = 1,048,576.
2. **Video needs the OpenAI Videos API.** LiteLLM never calls the sidecar's
   native `/v1/videos/generations`; it speaks the OpenAI Videos spec
   (`POST /videos` → poll `GET /videos/{id}` → `GET /videos/{id}/content`),
   which the sidecar exposes since 0.6.0.

Working `config.yaml` for all three:

```yaml
model_list:
  # --- images: flat per-image price (1024x1024) ---
  - model_name: grok-imagine-image
    litellm_params:
      model: openai/xai/grok-imagine-image
      api_base: http://localhost:4001/v1
      api_key: "dummy"
      input_cost_per_pixel: 1.9073486328125e-08   # $0.02 / 1048576 px
    model_info:
      mode: image_generation

  - model_name: grok-imagine-image-pro
    litellm_params:
      model: openai/xai/grok-imagine-image-pro
      api_base: http://localhost:4001/v1
      api_key: "dummy"
      input_cost_per_pixel: 6.67572021484375e-08  # $0.07 / 1048576 px
    model_info:
      mode: image_generation

  # --- video: $0.05/second ---
  - model_name: grok-imagine-video
    litellm_params:
      model: openai/xai/grok-imagine-video
      api_base: http://localhost:4001/v1
      api_key: "dummy"
      output_cost_per_second: 0.05
    model_info:
      mode: video_generation
```

Call them through LiteLLM:

```bash
# image
curl http://localhost:4000/v1/images/generations \
  -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
  -d '{"model": "grok-imagine-image", "prompt": "a corgi astronaut"}'

# video — create, then poll the returned id until status=completed
curl http://localhost:4000/v1/videos \
  -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
  -d '{"model": "grok-imagine-video", "prompt": "a corgi surfing", "seconds": "8"}'
```

Notes:

- Pass `seconds` on video creates — LiteLLM computes video spend from the
  `seconds` echoed on the create response (`output_cost_per_second ×
  seconds`), so omitting it records $0 for that call.
- Video jobs live in the sidecar process' memory (TTL 24h, override with
  `BLOCKRUN_VIDEO_JOB_TTL`); poll the same sidecar instance that accepted
  the create — don't run multiple sidecar replicas behind one LiteLLM
  video model without sticky routing.
- **Chat spend needs no config**: since 0.6.0 the sidecar returns the real
  x402 charge in the `x-litellm-response-cost` response header, which
  LiteLLM reads off openai-compatible upstreams and records as the
  request's spend — the exact wallet deduction, not an estimate.
- The custom-pricing numbers above are BlockRun's list prices; check
  [blockrun.ai/models](https://blockrun.ai/models) if they've moved.

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
| `POST` | `/v1beta/models/{model}:generateContent` | Native Gemini JSON request and response, with automatic x402 payment. **Wallet rail only** — 501 on the API-key rail. |
| `POST` | `/v1beta/models/{model}:streamGenerateContent` | Native Gemini SSE, with automatic x402 payment. **Wallet rail only** — 501 on the API-key rail. |
| `POST` | `/v1/responses` | OpenAI Responses API, bridged onto Chat Completions (`input`→`messages`, `output`/`response.*` SSE out). Text-in/text-out; for advanced tool/state flows use `/v1/chat/completions`. |
| `POST` | `/v1/images/generations` | OpenAI Image Generations. Accepts `prompt`, `model`, `size`, `n`, and `quality` (Solana or API key — see below). |
| `POST` | `/v1/images/edits` | OpenAI-compatible image editing. Accepts JSON data URIs or multipart `image`/`image[]`; supports multiple source images, `mask`, and `quality` (Solana only). `/v1/images/image2image` is an alias. |
| `POST` | `/v1/videos` | OpenAI Videos API create (what LiteLLM's video routes call) — returns a job object immediately |
| `GET`  | `/v1/videos/{id}` | OpenAI Videos API status poll (`queued` → `in_progress` → `completed`/`failed`) |
| `GET`  | `/v1/videos/{id}/content` | Download the finished clip bytes |
| `POST` | `/v1/videos/generations` | Native video generation. Supports `duration_seconds`, image/first-last-frame inputs, reference images, `input_type`, resolution, aspect ratio, audio, seed, and model-specific parameters. |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS |
| `POST` | `/v1/audio/generations` | Music generation |
| `POST` | `/v1/audio/sound-effects` | Cinematic sound effects |
| `GET`  | `/v1/models` | BlockRun model catalog |
| `GET`  | `/healthz` | Liveness probe (no upstream call) |
| `GET`  | `/docs` | Auto-generated Swagger UI |

### 2e. Native Gemini protocol

> **Wallet rail only.** `api.blockrun.ai` does not publish `/v1beta`, so with
> `BLOCKRUN_API_KEY` set the sidecar answers 501 with that explanation instead
> of a bare 404. Gemini *models* still work on the API-key rail through
> `/v1/chat/completions` with `model="google/gemini-3-pro"`; only Google's own
> protocol needs a wallet.

Native Gemini calls use the sidecar root (`http://localhost:4001`), not the
OpenAI `/v1` base. The sidecar preserves Gemini request/response JSON and SSE
frames and adds the x402 payment signature using the configured wallet.

```bash
# Non-streaming
curl http://localhost:4001/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Reply with native-ok"}]}]}'

# Streaming
curl -N 'http://localhost:4001/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse' \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Reply with stream-ok"}]}]}'
```

The official Google Gen AI Python SDK can use the same native protocol. Keep
the SDK itself; point its custom base URL at the sidecar and use any non-empty
placeholder API key (the sidecar removes it before forwarding):

```python
from google import genai
from google.genai import types

client = genai.Client(
    api_key="unused",
    http_options=types.HttpOptions(base_url="http://localhost:4001"),
)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Reply with native-ok",
)
print(response.text)

for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash-lite",
    contents="Reply with stream-ok",
):
    print(chunk.text or "", end="")
```

For Solana, start the sidecar with
`BLOCKRUN_API_URL=https://sol.blockrun.ai/api` and a `SOLANA_WALLET_KEY` (or
the wallet already stored at `~/.blockrun/.solana-session`). A client-supplied
Google API key is ignored; the local wallet is the authentication and payment
mechanism.

**Availability.** Native Gemini passthrough on the **Base** gateway (the
default, `https://blockrun.ai/api`) is limited to enterprise/allowlisted
wallets — other payers get `403 PERMISSION_DENIED` and are **not** charged. On
**Solana** (`https://sol.blockrun.ai/api`) it is generally available. If your
wallet is not on the Base allowlist, use Solana for native Gemini, or call the
OpenAI-format `/v1/chat/completions` route with `model="google/gemini-3.1-pro"`,
which works on both chains. Only `generateContent` and `streamGenerateContent`
are proxied; `countTokens`, model list/get, and `embedContent` are not served by
the gateway. Streaming always returns SSE regardless of `alt=` (the client query
string is dropped and the gateway sets `alt=sse` itself), so raw REST clients
must parse SSE, not the chunked-JSON-array form.

If you set `BLOCKRUN_PROXY_TOKEN` to guard the sidecar, the Google SDK cannot
send it (it only sends `x-goog-api-key`), so pass it explicitly as a Bearer
header: `types.HttpOptions(base_url=..., headers={"Authorization": "Bearer <token>"})`.

This is a sidecar HTTP feature. `litellm.completion(...)` remains the
OpenAI-compatible interface and continues to use `/v1/chat/completions`.

#### Image request limits

`n` is bounded to **1–10** locally. The Base `image2image` gateway schema has no
bound, so an out-of-range `n` would pass validation, take payment, and only then
fail at the provider — losing the prepaid USDC.

Multipart uploads are capped by `BLOCKRUN_MAX_IMAGE_BYTES` (default 12MB → 413)
and `BLOCKRUN_MAX_IMAGE_PARTS` (default 4 → 400; 4 is the most any model
accepts). Blank optional fields (`quality=`, `size=`, `model=`, `mask=`, `n=`)
mean "not set", matching what the gateway's own multipart handler does.

#### `quality` and `input_type` (requires `blockrun-llm>=1.7.0`)

**`quality`** (`low`/`medium`/`high`/`auto`, `openai/gpt-image-*`) is **Solana
only** — the Base gateway defines no such field. Sending it on Base still
returns **200** and generates the image, but the parameter is ignored and the
response carries an `x-blockrun-warning` header saying so (a warning is also
logged). Point `BLOCKRUN_API_URL` at the Solana gateway to make it take effect.

**`input_type`** (`text`/`image`/`first_last_frame`/`reference`) declares the
seed mode you intend on `/v1/videos/generations`. The gateway infers the mode
from the seed fields and returns **400 without charging** if your declaration
disagrees. Useful when seed fields are built dynamically: a dropped `image_url`
otherwise degrades silently to text-to-video and still bills you.

Reference-to-video (`reference_videos`/`reference_audios`) is **not** supported:
both gateways currently gate it off and would return 503.

### 2f. Image generation

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://localhost:4001/v1")
resp = client.images.generate(
    model="google/nano-banana",
    prompt="a corgi astronaut on the moon",
    size="1024x1024",
)
print(resp.data[0].url)  # always an HTTPS proxy URL
```

All image models return HTTPS proxy URLs — the BlockRun server handles any provider-level differences (e.g. base64 payloads) transparently before responding.

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

## Native fingerprint passthrough

BlockRun returns the **upstream provider's response verbatim** — no model
substitution, no response reshaping. The relay-detection signals a tool like
`cctest.ai` inspects therefore survive end to end, so the call looks like a
genuine **direct** upstream request rather than a proxied one:

| Signal | Where | Models |
|---|---|---|
| `system_fingerprint` (`fp_*`) | top-level response | OpenAI (`openai/gpt-5.5`, …) |
| `service_tier` | top-level response | OpenAI |
| `usage.prompt_tokens_details.cached_tokens` | usage block | OpenAI |
| `usage.cache_read_input_tokens` / `cache_creation_input_tokens` | usage block | Anthropic (`anthropic/claude-fable-5`, …) |
| `reasoning_content` | per message | reasoning models |

Both integration modes preserve these:

- **Proxy mode** returns the OpenAI-shaped JSON verbatim, so every field above
  is present on the wire.
- **Provider mode** keeps them on the `litellm.ModelResponse` (`response.system_fingerprint`,
  `response.usage`, `response.choices[0].message.reasoning_content`). In
  streaming, the lossy `GenericStreamingChunk` carries them on
  `provider_specific_fields`.

```python
resp = litellm.completion(model="blockrun/openai/gpt-5.5", messages=[...])
print(resp.system_fingerprint)   # e.g. "fp_abc123" — the real upstream value
```

> **Note:** Claude's native thinking-block `signature` is an Anthropic
> `/v1/messages`-only field. The litellm package speaks OpenAI
> `/v1/chat/completions`, so it surfaces `reasoning_content` and the cache-token
> usage but not the raw `signature`. For full Anthropic-native passthrough
> (content blocks + `signature`), use the `blockrun-llm-vip` `Anthropic` client.

Verified flagship models: **`openai/gpt-5.5`**, **`anthropic/claude-fable-5`**,
**`google/gemini-3.1-pro`**. The contract is locked by
`tests/test_fingerprint.py`.

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
latency_ms, stream, cost_usd, cost_source, estimated_cost_usd, settlement,
status, error_type, error_message, request_id
```

`cost_source` says how much to trust `cost_usd`:

| `cost_source` | Meaning |
|---|---|
| `blockrun_x402` | `cost_usd` **is** the settled on-chain charge for this call. Wallet rail. |
| `blockrun_account` | Billed to prepaid account credit — no per-call on-chain charge exists. `cost_usd` here is LiteLLM's token × list-price estimate (`null` when LiteLLM has no price for the model); the **authoritative figure is the ledger at [user.blockrun.ai](https://user.blockrun.ai) → Activity**. |
| `litellm_estimate` | Wallet rail, but no charge was reported (free/cached call, or an older SDK). The estimate is standing in. |

The distinction between the last two matters for reconciliation: `blockrun_account` means "a real number exists, elsewhere", not "we tried to read one and failed".

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
| `BLOCKRUN_API_KEY` (env) | BlockRun account key (`brk_live_…`) — issued at [user.blockrun.ai](https://user.blockrun.ai) | yes |
| `BLOCKRUN_API_BASE_URL` (env) | Account API endpoint (default `https://api.blockrun.ai`) | yes |
| `BLOCKRUN_CHAIN` (env) | Wallet-rail chain, `solana` (default) or `base` | yes |
| `SOLANA_WALLET_KEY` (env) | Solana private key | yes |
| `BLOCKRUN_WALLET_KEY` (env) | Base private key | yes |
| `~/.blockrun/.session` | Auto-created Base wallet | — |
| `~/.blockrun/.solana-session` | Auto-created Solana wallet | — |
| `~/.blockrun/litellm_calls.jsonl` | LiteLLM request log | `BLOCKRUN_LITELLM_LOG` env or `enable_local_logging(path)` |
| `~/.blockrun/cost_log.jsonl` | USDC cost audit for paid calls (SDK) | — |
| `~/.blockrun/data/*.json` | Full request/response archive for paid calls (SDK) | — |
| `BLOCKRUN_PROXY_TOKEN` (env) | Optional shared-secret guard on sidecar | yes |
| `BLOCKRUN_MAX_CONCURRENT` (env) | Max in-flight requests to upstream (default `100`) | yes |

---

## Examples

The `examples/` directory has copy-paste-ready snippets:

- [`examples/python_lib.py`](examples/python_lib.py) — full LiteLLM Python library usage
- [`examples/litellm_config.yaml`](examples/litellm_config.yaml) — LiteLLM Proxy Server config
- [`examples/raw_openai_sdk.py`](examples/raw_openai_sdk.py) — pointing the OpenAI SDK at the proxy
- [`examples/custom_callbacks.py`](examples/custom_callbacks.py) — JSONL log bridge for Proxy mode

---

## How it works (under the hood)

One adapter, two rails. The rail is chosen by which credential is present; nothing above that point differs.

**API-key rail** — a plain authenticated request:

```
┌─────────────────┐    OpenAI dict     ┌──────────────────────┐   Authorization: Bearer brk_…  ┌──────────────────┐
│ Your app /      │ ─────────────────▶ │  blockrun-litellm    │ ─────────────────────────────▶ │ api.blockrun.ai  │
│ LiteLLM /       │                    │  (provider OR proxy) │ ◀──── 200 + chat response ──── │  (your account)  │
│ OpenAI SDK      │                    └──────────────────────┘                                └────────┬─────────┘
└─────────────────┘                                                                                    │ debits
                                                                                                       │ prepaid
                                                                                                       ▼ credit
                                                                                              user.blockrun.ai
```

1. Caller sends an OpenAI Chat Completions dict.
2. `blockrun-litellm` whitelists the params and POSTs them with your key.
3. The account API meters real upstream token usage against the published price sheet and debits your credit.
4. Spend lands in the account ledger; the response comes back verbatim.

**Wallet rail** — x402, no account:

```
┌─────────────────┐    OpenAI dict     ┌──────────────────────┐    POST /v1/chat/completions  ┌────────────────┐
│ Your app /      │ ─────────────────▶ │  blockrun-litellm    │ ────────────────────────────▶ │ sol.blockrun.ai│
│ LiteLLM /       │                    │  (provider OR proxy) │ ◀──── 402 + payment-required ─│  (or blockrun. │
│ OpenAI SDK      │                    │  ↓                   │                               │   ai for Base) │
└─────────────────┘                    │  blockrun-llm SDK    │ ───── signed retry ─────────▶ │                │
                                       │  (local signing)     │ ◀──── 200 + chat response ────│                │
                                       └──────────────────────┘                               └────────────────┘
                                                ▲
                                                │ private key (stays local, signs only)
                                       ┌──────────────────────┐
                                       │ SOLANA_WALLET_KEY    │
                                       │   or ~/.blockrun/    │
                                       └──────────────────────┘
```

1. Caller sends an OpenAI Chat Completions dict.
2. `blockrun-litellm` whitelists the params and dispatches through `blockrun-llm`.
3. `blockrun-llm` posts to BlockRun, receives a 402 with payment requirements, signs the payment locally with your wallet (SVM on Solana, EIP-712 on Base), and retries.
4. BlockRun verifies the signature on-chain, settles the USDC micropayment, runs the inference, and returns the response — plus the exact charge, which the adapter surfaces as `cost_usd`.
5. `blockrun-litellm` returns the dumped pydantic model as a plain OpenAI dict (or `litellm.ModelResponse` in provider mode).

---

## FAQ

**Q: Does this support streaming?**
Yes, as of v0.2.0. Pass `stream=True` and the adapter routes through `blockrun-llm`'s `chat_completion_stream()` (SDK ≥ 0.20.0). The 402 → sign-locally → retry-with-PAYMENT-SIGNATURE dance happens before the first chunk; once the upstream switches to `text/event-stream`, chunks are forwarded straight through (provider mode → `litellm.GenericStreamingChunk`, proxy mode → OpenAI-style `data: <json>\n\n` SSE). Caveats inherited from the gateway: `search_parameters` and the Responses-API models (`codex`, `gpt-5.4-pro`) reject streaming server-side with 400.

**Q: Do I need a crypto wallet to use this?**
No. Sign in at [user.blockrun.ai](https://user.blockrun.ai), top up by card, and set `BLOCKRUN_API_KEY`. The wallet rail stays available for anyone who prefers to pay in USDC directly — including agents, which can hold a wallet but cannot fill in a card form.

**Q: I already use this with a wallet. Does the API key change anything for me?**
No. The wallet rail is untouched; the account rail only activates when a `brk_`-prefixed key is present. The one behaviour change in 0.10.0 is the default chain — see the next question.

**Q: How do I switch between Solana and Base?**
`BLOCKRUN_CHAIN=solana` (the default since 0.10.0) or `BLOCKRUN_CHAIN=base`; `--chain` on the sidecar; or point `BLOCKRUN_API_URL` / `api_base=` at a gateway directly, which wins over both. A host that holds only a Base wallet and sets nothing keeps using Base, with a warning.

**Q: Where does my private key live?**
On your machine only — `SOLANA_WALLET_KEY` / `BLOCKRUN_WALLET_KEY` env vars, or `~/.blockrun/.solana-session` / `~/.blockrun/.session` if you used `setup_agent_wallet()`. The proxy and provider both read from those sources via `blockrun-llm`. Only signatures are transmitted. On the API-key rail there is no private key at all.

**Q: Why is `cost_usd` empty when I use an API key?**
Because there is no per-call on-chain charge to report — the call was billed against prepaid credit. The audit row says so with `cost_source: "blockrun_account"`, and the authoritative spend is at [user.blockrun.ai](https://user.blockrun.ai) → Activity. The wallet rail still reports the exact settled charge per call.

**Q: Can I run the proxy in Docker / k8s?**
Yes — it's a vanilla FastAPI app. Pass `BLOCKRUN_API_KEY` (or the wallet key) via secret, bind to `0.0.0.0` only inside a private network, and set `BLOCKRUN_PROXY_TOKEN` for an additional auth layer. The sidecar never forwards a client's `Authorization` header upstream.

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

[BlockRun](https://blockrun.ai) 的 [LiteLLM](https://github.com/BerriAI/litellm) 适配层 —— 用 LiteLLM 调用 BlockRun 上 90+ 个 AI 模型，**完全零改动**。可以用 **BlockRun API Key**（信用卡充值，不需要钱包），也可以用 **x402 USDC 钱包**（**Solana** 或 Base）。

> **一句话：** BlockRun 的 `/v1/chat/completions` 协议层就是 OpenAI 兼容的，区别只在*怎么付钱*。两条路：一条是普通 API Key，从预付余额扣；一条是按次 x402 钱包签名（非托管 USDC，Solana / Base）。这个包两条都支持，凭证之上的一切完全一致。

## 领 API Key（30 秒）

1. 用 Google 账号登录 **[user.blockrun.ai](https://user.blockrun.ai)**。
2. **Billing → Add credit**：信用卡充值，最低 $5。手续费（5.5% + $0.30）在**充值时**一次性收取，之后每个模型都按官网标价计费 —— 没有每次调用的最低消费，没有每次调用的手续费，没有加价。
3. **API keys → Create key**：拿到 `brk_live_…`，只显示一次。

```bash
export BLOCKRUN_API_KEY=brk_live_...
```

到此配置就结束了。不需要钱包、不需要链、不需要 USDC、不需要 gas。

想用自己的钱包付？看下面[**用 x402 钱包付费**](#用-x402-钱包付费solana--base) —— 那条路完全不需要注册账号。

## 两种付费方式

|  | **API Key** | **x402 钱包** |
|---|---|---|
| 怎么开通 | 登录 [user.blockrun.ai](https://user.blockrun.ai) 刷卡充值 | 给钱包充 USDC |
| 凭证 | `BLOCKRUN_API_KEY=brk_live_…` | `SOLANA_WALLET_KEY` / `BLOCKRUN_WALLET_KEY` |
| 端点 | `https://api.blockrun.ai` | `https://sol.blockrun.ai/api`（默认）或 `https://blockrun.ai/api` |
| 计费 | 预付余额，按标价扣 | 每次调用链上结算 USDC |
| 要不要账号 | 要 | **不要** |
| 链 | 没有链 | Solana 或 Base |
| 单次成本上报 | 没有 —— 查 [user.blockrun.ai](https://user.blockrun.ai) 账单 | **有** —— 每次调用返回真实结算金额 |
| 消费在哪看 | Dashboard → Activity | 链上，以及 `x-blockrun-settlement` |
| 原生 Gemini (`/v1beta`) | 不支持 | 支持 |
| 需要装的 extra | 无 | Solana 签名需要 `[solana]` |

其余完全相同：同一份模型目录、同样的 OpenAI / Anthropic 协议、同样的流式、同样的原生指纹透传。

**优先级：** 只要检测到 API Key 就走 Key 这条路。`BLOCKRUN_API_KEY`（或 `--api-key`，或调用时传 `api_key="brk_live_…"`）选账号路；没有 Key 就回落到钱包路。钱包私钥不会被误判成 API Key —— 只有 `brk_` 前缀才会切到账号路，而任何私钥格式都不以它开头。

## 两种对接方式

| 模式 | 适用 | 写法 |
|---|---|---|
| **1. 自定义 Provider**（进程内） | 用 LiteLLM **Python 库**的应用 | `litellm.completion(model="blockrun/openai/gpt-5.5", ...)` |
| **2. 本地代理**（sidecar） | 用 LiteLLM **Proxy Server** 的、或任何 OpenAI 客户端 | `api_base="http://localhost:4001/v1"` |

两种模式在两条付费路上都能用，行为一致。按你的部署方式选一种就行。

## 快速上手

### 安装

```bash
# API Key，或在 Python 库里用 Solana / Base 钱包
pip install blockrun-litellm

# 再加本地代理（FastAPI/uvicorn）
pip install 'blockrun-litellm[proxy]'

# 再加 x402 SVM 签名器 —— 只有用 Solana 钱包付费才需要
pip install 'blockrun-litellm[proxy,solana]'
```

用 API Key 的话不需要 `solana` extra —— 那条路没有签名这一步。

### 用 API Key（推荐先试这个）

```python
import litellm
from blockrun_litellm import register

register()

# 从环境变量读 BLOCKRUN_API_KEY；也可以每次调用传 api_key=
r = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=64,
)
print(r.choices[0].message.content)
```

sidecar 版：

```bash
blockrun-litellm-proxy --port 4001 --api-key brk_live_...
```

## 用 x402 钱包付费（Solana / Base）

不用注册、不用账号：给钱包充 USDC，每次请求自己结算。这是给 Agent 用的那条路 —— Agent 可以持有钱包，但填不了信用卡表单。

| 链 | 网关 URL | 钱包环境变量 | 说明 |
|---|---|---|---|
| **Solana (USDC)** —— *默认* | `https://sol.blockrun.ai/api` | `SOLANA_WALLET_KEY` | 亚秒级结算、费用最低。需要 `[solana]` extra。同步 / 异步 / 流式都支持。 |
| Base (USDC) | `https://blockrun.ai/api` | `BLOCKRUN_WALLET_KEY` | 同步 / 异步 / 流式都支持。 |

**从 0.10.0 起 Solana 是默认链**（之前没配置时默认走 Base）。显式指定：

```bash
export BLOCKRUN_CHAIN=solana   # 默认
export BLOCKRUN_CHAIN=base
```

也可以直接用 `BLOCKRUN_API_URL` / `--api-url` / `api_base=` 指到具体网关（优先级高于 `BLOCKRUN_CHAIN`）。

> **从 ≤ 0.9.x 升级、原来用 Base 的？** 不会坏。没有配置链、且机器上只有 Base 钱包时，适配器仍然走 Base，并打一行警告。设 `BLOCKRUN_CHAIN=base` 把选择写死，警告就没了。

### 配钱包（一次性）

```bash
# 方式 A — 环境变量（服务端推荐）
export SOLANA_WALLET_KEY=YOUR_SOLANA_PRIVATE_KEY      # Solana（默认链）
export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_PRIVATE_KEY    # Base

# 方式 B — 自动创建并扫码充值（交互式）
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"
```

私钥**只在本地签名**，永远不会离开你的机器。

> 💡 想零成本试一遍？用免费模型 `nvidia/deepseek-v4-flash` —— 代码完全一样，钱包流程一样，结算 $0。

### 两条路各自支持哪些接口

除下面注明的一条外，所有接口两条路都支持：`/v1/chat/completions`（含流式）、`/v1/messages`（Anthropic 原生）、`/v1/responses`、`/v1/images/*`、`/v1/videos*`、`/v1/audio/*`、`/v1/models`。

唯一的缺口是**原生 Gemini 协议** `/v1beta/models/{model}:generateContent` —— `api.blockrun.ai` 没有发布这个接口，用 API Key 时 sidecar 返回 501 并说明原因。Gemini **模型本身**两条路都能用，走 `/v1/chat/completions` 传 `model="google/gemini-3-pro"` 即可；只有 Google 自家协议需要钱包。

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
# 1) 启动 sidecar —— 二选一
blockrun-litellm-proxy --port 4001 --api-key brk_live_...   # API Key
export SOLANA_WALLET_KEY=YOUR_SOLANA_PRIVATE_KEY            # 或 x402 钱包（默认 Solana）
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

### 图像 / 视频模型上 LiteLLM：调用 + 计费

把 `xai/grok-imagine-image` / `-image-pro` / `grok-imagine-video` 挂到 LiteLLM Proxy 时有两个坑：

1. **LiteLLM 记账为 $0** —— LiteLLM 按自带价格表算钱，表里没有这些模型，成本查询失败就记 0。解法：在 `litellm_params` 里声明[自定义单价](https://docs.litellm.ai/docs/proxy/custom_pricing)。图像按 `input_cost_per_pixel × 宽 × 高 × n` 计，固定张价除以 1024×1024。
2. **视频要走 OpenAI Videos API** —— LiteLLM 不会调 sidecar 原生的 `/v1/videos/generations`，它打的是 `POST /videos` → 轮询 `GET /videos/{id}` → `GET /videos/{id}/content`，sidecar 0.6.0 起已支持。

```yaml
model_list:
  - model_name: grok-imagine-image
    litellm_params:
      model: openai/xai/grok-imagine-image
      api_base: http://localhost:4001/v1
      api_key: "dummy"
      input_cost_per_pixel: 1.9073486328125e-08   # $0.02/张 ÷ 1048576 像素
    model_info:
      mode: image_generation

  - model_name: grok-imagine-image-pro
    litellm_params:
      model: openai/xai/grok-imagine-image-pro
      api_base: http://localhost:4001/v1
      api_key: "dummy"
      input_cost_per_pixel: 6.67572021484375e-08  # $0.07/张 ÷ 1048576 像素
    model_info:
      mode: image_generation

  - model_name: grok-imagine-video
    litellm_params:
      model: openai/xai/grok-imagine-video
      api_base: http://localhost:4001/v1
      api_key: "dummy"
      output_cost_per_second: 0.05                # $0.05/秒
    model_info:
      mode: video_generation
```

注意事项：

- 视频创建请求**务必带 `seconds`**（如 `"8"`）—— LiteLLM 按创建响应回显的 seconds × 每秒单价计费，不带就记 $0。
- 视频任务存在 sidecar 进程内存里（TTL 24 小时，`BLOCKRUN_VIDEO_JOB_TTL` 可调），轮询要打到接收创建请求的同一个 sidecar 实例。
- **Chat 计费无需任何配置**：0.6.0 起 sidecar 在响应头 `x-litellm-response-cost` 返回真实 x402 扣费，LiteLLM 直接采用，分毫不差。

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

**Q：一定要有加密钱包吗？**
不用。去 [user.blockrun.ai](https://user.blockrun.ai) 登录、刷卡充值、设 `BLOCKRUN_API_KEY` 就行。钱包那条路继续保留，给愿意直接用 USDC 付费的人 —— 尤其是 Agent，它能持有钱包，但填不了信用卡表单。

**Q：我已经在用钱包，加了 API Key 会影响我吗？**
不会。钱包那条路一行没动，只有出现 `brk_` 前缀的凭证时才会切到账号路。0.10.0 唯一的行为变化是默认链，见下一条。

**Q：怎么在 Solana 和 Base 之间切？**
`BLOCKRUN_CHAIN=solana`（0.10.0 起是默认）或 `BLOCKRUN_CHAIN=base`；sidecar 用 `--chain`；也可以直接把 `BLOCKRUN_API_URL` / `api_base=` 指到具体网关（优先级最高）。机器上只有 Base 钱包又什么都没配的，仍然走 Base，并打一行警告。

**Q：私钥放哪？**
只在本地 —— `SOLANA_WALLET_KEY` / `BLOCKRUN_WALLET_KEY` 环境变量，或 `setup_agent_wallet()` 创建的 `~/.blockrun/.solana-session` / `~/.blockrun/.session`。Provider 和 Proxy 都通过 `blockrun-llm` 读取。链上只看到签名，看不到私钥。用 API Key 时根本不存在私钥。

**Q：用 API Key 时 `cost_usd` 为什么是空的？**
因为这条路没有"单次链上扣款"这回事 —— 这次调用是从预付余额扣的。审计行会写明 `cost_source: "blockrun_account"`，权威金额在 [user.blockrun.ai](https://user.blockrun.ai) → Activity。钱包那条路仍然每次返回真实结算金额。

**Q：Docker / k8s 部署？**
代理是普通的 FastAPI 应用。`BLOCKRUN_API_KEY`（或钱包私钥）用 secret 注入，对外只暴露内网，可选 `BLOCKRUN_PROXY_TOKEN` 加一层 Bearer 鉴权。sidecar 不会把客户端的 `Authorization` 头转发到上游。

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
