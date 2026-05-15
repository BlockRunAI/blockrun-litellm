# BlockRun × LiteLLM — 5-minute onboarding for customers

> Verified end-to-end on 2026-05-15 with `litellm 1.83.x`,
> `blockrun-litellm 0.3.5`, `blockrun-llm 0.24.1`, Python 3.13.
> Every command below was actually run; the outputs are real.
>
> **Two chains, two integration modes:**
>
> | Chain | Gateway URL | Wallet env var | Recommended? |
> |---|---|---|---|
> | **Solana** (USDC) | `https://sol.blockrun.ai/api` | `SOLANA_WALLET_KEY` | ⭐ **Default** — measured median settlement **106 ms** (Base 935 ms, **8.8× faster**), gas < $0.001/call |
> | Base (USDC) | `https://blockrun.ai/api` *(default URL)* | `BLOCKRUN_WALLET_KEY` | 99.9% success rate (vs Solana 94.7%) — use as Solana's fallback |
>
> | Your setup | Use mode |
> |---|---|
> | You call `litellm.completion(...)` from Python code | **Mode A — Python library** |
> | You run `litellm --config ...` as a central proxy | **Mode B — LiteLLM Proxy Server** |

---

## Quick decision tree

```
Are you running LiteLLM Proxy Server centrally?
├─ YES → Mode B
└─ NO  → Mode A

Which chain do you want to pay in?
├─ Solana USDC ⭐ recommended (8.8× faster settle, sub-cent gas)
│                                          → SOLANA_WALLET_KEY env var
│                                          + api_base="https://sol.blockrun.ai/api"
└─ Base USDC (99.9% success rate; good as Solana's fallback)
                                          → BLOCKRUN_WALLET_KEY env var
```

---

## Install (both modes)

```bash
# Base only — minimal
pip install -U 'blockrun-litellm[proxy]'

# Base + Solana — adds the x402 SVM toolchain
pip install -U 'blockrun-litellm[proxy,solana]'
```

Python ≥ 3.9. (Tested with 3.13.)

---

## Wallet setup (both modes, one-time)

You only need a wallet **on the chain you want to use**. You can use both.

### Base

```bash
# Option A — already have a Base private key
export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_CHAIN_PRIVATE_KEY

# Option B — auto-create one + show QR for USDC funding
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"
# Wallet key saved to ~/.blockrun/.session; auto-loaded next time.
```

### Solana

```bash
# Option A — already have a Solana base58 secret key
export SOLANA_WALLET_KEY=YOUR_BASE58_SOLANA_SECRET_KEY

# Option B — auto-create + show QR for USDC funding (Solana mainnet)
python -c "from blockrun_llm import setup_agent_solana_wallet; setup_agent_solana_wallet()"
# Key saved to ~/.blockrun/.solana-session
```

> 💡 The private key **never leaves your machine** — it's used only to sign EIP-712 (Base) or Ed25519 (Solana) payment payloads locally. Only the signature is sent over the wire. Same security model as MetaMask / Phantom.

> 💡 Want to test without spending USDC? Use a free model like `nvidia/deepseek-v4-flash` — same code path, zero settlement, works on both chains.

---

## Mode A — Python library

### Base (default)

```python
import litellm
from blockrun_litellm import register
register()  # idempotent; adds "blockrun" to litellm.custom_provider_map

# Non-streaming
response = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(response.choices[0].message.content)

# Streaming
for chunk in litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Solana — pass `api_base`

```python
import litellm
from blockrun_litellm import register
register()

for chunk in litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    api_base="https://sol.blockrun.ai/api",   # ← decides chain
    # api_key="..."                            # optional, else SOLANA_WALLET_KEY env var
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

**Verified output (Solana, 2026-05-12):** `"Hello! How can I"` — 2 chunks, ~1s.

**Async + Solana (since `blockrun-litellm 0.3.1` / `blockrun-llm 0.22.0`):** `litellm.acompletion(stream=True)` works the same way — pass `api_base="https://sol.blockrun.ai/api"` and the adapter routes to the async Solana client.

```python
async for chunk in await litellm.acompletion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    api_base="https://sol.blockrun.ai/api",
    api_key="solana-private-key",
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

---

## Mode B — LiteLLM Proxy Server

```
┌────────────┐    ┌────────────┐     ┌─────────────────┐     ┌──────────────────────────────┐
│ your app   │───▶│ LiteLLM    │────▶│ blockrun-       │────▶│ blockrun.ai (Base)           │
│ (any lang) │    │ Proxy      │     │ litellm-proxy   │     │ OR sol.blockrun.ai (Solana)  │
│            │    │ (port 4000)│     │ (port 4001)     │     │                              │
└────────────┘    └────────────┘     └─────────────────┘     └──────────────────────────────┘
                                              │
                                              │ signs payment locally
                                              ▼
                                       ~/.blockrun/.session       ← Base
                                       ~/.blockrun/.solana-session ← Solana
```

### 1. Start the sidecar (terminal 1)

```bash
# Base chain (default)
export BLOCKRUN_WALLET_KEY=0xYOUR_KEY
blockrun-litellm-proxy --port 4001

# Solana chain
export SOLANA_WALLET_KEY=YOUR_SOLANA_KEY
blockrun-litellm-proxy --port 4001 --api-url https://sol.blockrun.ai/api
```

The sidecar is **single-wallet per process**. For multi-tenant (each customer with their own wallet), run one sidecar per tenant on different ports.

### 2. Wire it into `config.yaml`

```yaml
model_list:
  # Base model
  - model_name: gpt-5.5-base
    litellm_params:
      model: openai/openai/gpt-5.5
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

  # Solana model (point at a different sidecar on a different port)
  # - model_name: gpt-5.5-solana
  #   litellm_params:
  #     model: openai/openai/gpt-5.5
  #     api_base: http://127.0.0.1:4002/v1
  #     api_key: "dummy"

  # Free model — works on both chains, $0 settlement
  - model_name: deepseek-v4-flash
    litellm_params:
      model: openai/nvidia/deepseek-v4-flash
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

litellm_settings:
  drop_params: True             # silently drops OpenAI params BlockRun ignores
  callbacks: ["custom_callbacks.blockrun_logger"]   # optional: see "Local request log" below

general_settings:
  master_key: "sk-your-litellm-master-key"
```

### 3. Start LiteLLM Proxy (terminal 2)

```bash
litellm --config config.yaml --port 4000 --host 127.0.0.1
```

### 4. Use any OpenAI client

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"Say one word"}],
    "stream": true
  }'
```

**Verified output (Mode B, 2026-05-12):** HTTP 200, `text/event-stream`, content `"Okay."`.

---

## Local request log (input/output tokens, latency, cost)

`blockrun-litellm 0.2.2+` ships a one-line opt-in JSONL logger.

### Where the log file lives

| Source | Default path | Override |
|---|---|---|
| Explicit arg to `enable_local_logging("...")` | (whatever you pass) | — |
| `BLOCKRUN_LITELLM_LOG` env var | (whatever it points to) | — |
| Otherwise | **`~/.blockrun/litellm_calls.jsonl`** | one of the above |

The directory is created automatically; the file is appended to forever (no rotation — pipe through `logrotate` or truncate periodically). One JSON object per line.

### What each row contains

```
ts, iso, model, provider, messages, completion, usage{prompt_tokens,
completion_tokens, total_tokens}, latency_ms, stream, cost_usd, status,
error_type, error_message, request_id
```

### Mode A — one line

```python
from blockrun_litellm import enable_local_logging
enable_local_logging()                       # ~/.blockrun/litellm_calls.jsonl
# or enable_local_logging("/var/log/blockrun.jsonl")
```

### Mode B — drop a bridge file

LiteLLM Proxy loads callbacks **by filename**, not by installed package, so add `custom_callbacks.py` next to your `config.yaml`:

```python
# custom_callbacks.py
from blockrun_litellm.logger import JSONLLogger
blockrun_logger = JSONLLogger()
```

Then in `config.yaml`:

```yaml
litellm_settings:
  callbacks: ["custom_callbacks.blockrun_logger"]
```

`BLOCKRUN_LITELLM_LOG` env var works in both modes.

### Sample row (real, verified)

```json
{
  "ts": 1778566732.452,
  "iso": "2026-05-12T01:38:52",
  "model": "nvidia/deepseek-v4-flash",
  "provider": "nvidia",
  "messages": [{"role": "user", "content": "Say one word"}],
  "completion": "Serendipity",
  "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
  "latency_ms": 4128.3,
  "stream": true,
  "cost_usd": null,
  "status": "success"
}
```

---

## All the persistent files in one table

| File / path | What's there | Configurable? |
|---|---|---|
| `$BLOCKRUN_WALLET_KEY` / `$BASE_CHAIN_WALLET_KEY` (env) | Base private key (preferred for prod) | yes |
| `$SOLANA_WALLET_KEY` (env) | Solana private key (preferred for prod) | yes |
| `~/.blockrun/.session` | Auto-created Base wallet | — |
| `~/.blockrun/.solana-session` | Auto-created Solana wallet | — |
| `~/.blockrun/litellm_calls.jsonl` | LiteLLM request log (this adapter) | `BLOCKRUN_LITELLM_LOG` env or arg |
| `~/.blockrun/cost_log.jsonl` | Per-paid-call USDC cost audit (`blockrun-llm` SDK) | not configurable |
| `~/.blockrun/data/*.json` | Full request/response archive for paid calls | not configurable |
| `$BLOCKRUN_PROXY_TOKEN` (env) | Optional shared-secret guard on the sidecar | yes |

---

## What about errors?

`blockrun-litellm` translates transient upstream errors to LiteLLM's retriable exceptions so `litellm.Router` does the right thing:

| BlockRun upstream error | What your app sees |
|---|---|
| 503 Service Unavailable | `litellm.ServiceUnavailableError` |
| 502 Bad Gateway, 504 Gateway Timeout | `litellm.APIConnectionError` |
| 500 Internal Server Error | `litellm.InternalServerError` |
| 429 Too Many Requests | `litellm.RateLimitError` |
| Read/connect timeout | `litellm.Timeout` |

So your existing LiteLLM router config — `num_retries`, `fallbacks`, `cooldown_time` — just works:

```yaml
router_settings:
  num_retries: 2
  fallbacks:
    - {"gpt-5.5-base": ["claude-opus-4-7"]}
```

Plus the SDK itself retries 5xx in-band 3 times with exponential backoff (1s / 2s / 4s) before bubbling the error up.

For an extra-resilient call, you can also tell the SDK to walk a fallback list **before** the first chunk:

```python
litellm.completion(
    model="blockrun/nvidia/deepseek-v4-flash",
    messages=[...],
    fallback_models=["nvidia/llama-4-maverick"],   # picked up by the BlockRun SDK
)
```

---

## What about streaming?

✅ Full support since `blockrun-litellm 0.2.0` (Base) and `0.3.0` (Solana). SSE chunks pass through cleanly (`text/event-stream` with `data: <json>\n\n` and a terminating `data: [DONE]`).

**Caveats inherited from the BlockRun gateway** (returns HTTP 400 server-side):
- Models using OpenAI's Responses API (`codex`, `gpt-5.4-pro`) don't stream.
- xAI Live Search (`search_parameters`) is non-streaming.

**Solana async:** supported since `blockrun-litellm 0.3.1` / `blockrun-llm 0.22.0` (`AsyncSolanaLLMClient`). Earlier versions raised `NotImplementedError` — upgrade if you hit that.

---

## Frequently-asked

### Q: Do I need an OpenAI / Anthropic / Google API key?
No. BlockRun is a unified gateway — one wallet pays for all of them. Your existing per-vendor keys are not used here.

### Q: How much does each call cost?
- **Free tier:** $0 for `nvidia/deepseek-v4-flash`, `nvidia/llama-4-maverick`, etc.
- **Paid:** Per-token pricing per model; see `/v1/models` or [blockrun.ai](https://blockrun.ai). Typical Claude Opus 4 call is ~$0.001–0.01.

The cost per paid call lives in `~/.blockrun/cost_log.jsonl`. Aggregate it with:

```python
from blockrun_llm import get_cost_log_summary
print(get_cost_log_summary(group_by="model"))
```

### Q: What if my wallet runs out of USDC?
The gateway returns `402 Payment Required`. With router fallbacks (above), LiteLLM picks the next provider. Without fallbacks, your app gets `litellm.AuthenticationError` and you top up.

### Q: How do I run this in Docker / k8s?
Standard FastAPI app:

```dockerfile
FROM python:3.13-slim
RUN pip install 'blockrun-litellm[proxy,solana]'
# Inject keys via secret manager — never bake them into the image:
ENV BLOCKRUN_WALLET_KEY=""
ENV SOLANA_WALLET_KEY=""
CMD ["blockrun-litellm-proxy", "--host", "0.0.0.0", "--port", "4001"]
```

For shared deployments, set `BLOCKRUN_PROXY_TOKEN=...` and have clients send `Authorization: Bearer $TOKEN`.

### Q: How do I see which models are available?
```bash
curl http://127.0.0.1:4001/v1/models      # via sidecar
curl https://blockrun.ai/api/v1/models    # direct, Base
curl https://sol.blockrun.ai/api/v1/models # direct, Solana
```

### Q: Should I pick Base or Solana?

**Solana, strongly recommended.** Based on the last 3 days (2026-05-10 → 05-12) of production settlement records — **4,424 real on-chain payments**:

| Metric | **Solana** (PayAI facilitator) | Base (Coinbase CDP) | Solana advantage |
|---|---|---|---|
| **Median settlement** | **106 ms** ⚡ | 935 ms | **8.8× faster** |
| p90 settlement | 133 ms | 1,658 ms | 12.5× faster |
| p99 settlement | 178 ms | 2,577 ms | 14.5× faster |
| Success rate | 94.7% | 99.9% | Base more reliable |
| Failure time (median) | 154 ms (fast-fail) | 17.8 sec (gas-estimation retry storm) | Solana fails fast |

**Where you feel it most: streaming "time to first chunk":**
- **Solana**: ~106ms settle + upstream warm-up ≈ **~150ms to first chunk**
- **Base**: ~935ms settle + upstream warm-up ≈ **~1.1s to first chunk**

**Verdict:**
- ✅ **Solana** — sub-cent gas (< $0.001/call), first chunk 9× faster, full feature parity with Base (tool calling, async streaming since `blockrun-litellm 0.3.2`)
- ✅ **Base** — slightly more reliable (99.9% vs 94.7%), deeper USDC liquidity. Good as a fallback when Solana settlement fails.

**Recommended pattern**: route Solana by default with a router fallback to Base for the rare failure:

```yaml
router_settings:
  num_retries: 2
  fallbacks:
    - {"blockrun-solana-*": ["blockrun-base-*"]}
```

Switching between chains is a single-line change to `api_base` and `api_key` — your business code stays the same. (Solana's image / Exa / Predexon endpoints are sync-only today; chat is fully async + streaming.)

---

## Links

- 📦 PyPI: https://pypi.org/project/blockrun-litellm/
- 🌐 GitHub: https://github.com/BlockRunAI/blockrun-litellm
- 🛠 Underlying SDK: https://github.com/BlockRunAI/blockrun-llm
- 🌐 BlockRun docs: https://blockrun.ai

Questions? File a GitHub issue at https://github.com/BlockRunAI/blockrun-litellm/issues — replies typically within 24 hours.
