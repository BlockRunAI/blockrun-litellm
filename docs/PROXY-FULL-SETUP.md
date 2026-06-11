# LiteLLM Proxy + BlockRun — full setup (UI included)

> Verified end-to-end on 2026-05-15, macOS, Python 3.13, Docker Desktop
> 28.4, LiteLLM 1.83.x, blockrun-litellm 0.3.5, blockrun-llm 0.24.1.
> Every command below was actually run by a human and produced the
> output shown.
>
> This is the **full** setup: LiteLLM Proxy Server with the admin UI,
> Postgres for metadata/spend/keys, the BlockRun sidecar for x402
> payment signing, and a JSONL request log. Pick this guide when you
> want the LiteLLM dashboard at `http://localhost:4000/ui`.
>
> If you only need API access (no UI, no DB), see
> [`CUSTOMER-ONBOARDING.md`](CUSTOMER-ONBOARDING.md) — simpler and
> doesn't need Docker.

---

## What you'll get

After 10 minutes you'll have:

```
┌────────────┐    HTTPS API     ┌────────────────────┐    OpenAI-compat HTTP    ┌──────────────────┐    HTTPS + x402    ┌─────────────────────┐
│ your app   │─────────────────▶│ LiteLLM Proxy       │─────────────────────────▶│ blockrun-litellm │───────────────────▶│ blockrun.ai (Base)  │
│ or curl    │  Bearer key      │ + Admin UI          │   api_base=http://4001   │ -proxy (sidecar) │   PAYMENT-SIGNATURE│ OR sol.blockrun.ai  │
│            │                  │ port 4000           │                          │ port 4001        │                    │ (Solana)            │
└────────────┘                  └────────────────────┘                          └──────────────────┘                    └─────────────────────┘
                                          │                                              │
                                          │ Prisma                                       │ EIP-712 / Ed25519 sign locally
                                          ▼                                              ▼
                                  ┌──────────────────┐                          ┌──────────────────────┐
                                  │ Postgres 16      │                          │ $BLOCKRUN_WALLET_KEY │
                                  │ port 5544 docker │                          │ $SOLANA_WALLET_KEY   │
                                  │ vol litellm-pg   │                          │ ~/.blockrun/.session │
                                  └──────────────────┘                          └──────────────────────┘
```

**UI shows:** model list, request logs, per-request usage, spend per
key, virtual keys (sub-tokens with budgets / model allowlists / rate
limits), API key admin, settings.

---

## Prerequisites

| Need | Why |
|---|---|
| Python ≥ 3.9 (3.13 tested) | Run LiteLLM and the sidecar |
| Docker Desktop | Postgres for LiteLLM Proxy UI |
| A Base or Solana wallet with a few USDC | Pay-per-request |
| Port 4000, 4001 free on localhost | Proxy and sidecar |
| Port 5544 (or any free port) for Postgres | The default 5432 is usually taken by Postgres.app etc. — pick something free |

---

## Step 1 — Install the Python packages

```bash
python3.13 -m venv ~/.venv/litellm-blockrun
source ~/.venv/litellm-blockrun/bin/activate

pip install -U 'litellm[proxy]' 'blockrun-litellm[proxy,solana]' prisma
```

The `prisma` package is **required** for LiteLLM Proxy's UI / DB
features — `litellm[proxy]` does NOT pull it in automatically.
`blockrun-litellm[proxy,solana]` brings the BlockRun adapter, the
sidecar, and the x402 SVM toolchain for the Solana chain.

---

## Step 2 — Set up your BlockRun wallet (one-time)

You only need a wallet **on the chain you'll pay in**.

### Base (USDC on Base)

```bash
# A. Already have a Base private key:
export BLOCKRUN_WALLET_KEY=0xYOUR_BASE_CHAIN_PRIVATE_KEY

# B. Auto-create + show QR for USDC funding:
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"
# → saves to ~/.blockrun/.session ; auto-loaded next time
```

### Solana (USDC on Solana — cheaper gas)

```bash
# A. Existing Solana base58 secret key:
export SOLANA_WALLET_KEY=YOUR_BASE58_SOLANA_SECRET_KEY

# B. Auto-create + QR for funding:
python -c "from blockrun_llm import setup_agent_solana_wallet; setup_agent_solana_wallet()"
# → saves to ~/.blockrun/.solana-session
```

> The private key **never leaves your machine** — only EIP-712 (Base)
> or Ed25519 (Solana) signatures are transmitted.

> 💡 Want to test with $0 cost? Use the free model
> `nvidia/deepseek-v4-flash` — same flow, zero settlement.

---

## Step 3 — Start the BlockRun sidecar

In **terminal 1**:

```bash
source ~/.venv/litellm-blockrun/bin/activate
export BLOCKRUN_WALLET_KEY=...           # if Base
# export SOLANA_WALLET_KEY=...           # if Solana

blockrun-litellm-proxy --port 4001
# → INFO:     Uvicorn running on http://127.0.0.1:4001
```

For Solana, also pass the gateway URL:

```bash
blockrun-litellm-proxy --port 4001 --api-url https://sol.blockrun.ai/api
```

Verify:

```bash
curl http://127.0.0.1:4001/healthz
# → {"status":"ok"}
```

---

## Step 4 — Start Postgres for LiteLLM's UI

The LiteLLM admin UI **requires** a Postgres database — without it,
login fails with `Authentication Error, Not connected to DB!`. Run a
local one in Docker.

**⚠️ Watch out for port 5432** — many Macs already have
Postgres.app or Homebrew postgres on 5432, which silently shadows
Docker's port mapping (host's psql connects to the wrong server).
Use a free port like **5544** instead.

```bash
# Check 5432 is taken (you'll usually see Postgres.app here):
lsof -nP -iTCP:5432 | grep LISTEN

# Start a fresh Postgres on 5544:
docker run -d --name litellm-pg \
  -e POSTGRES_PASSWORD=litellm \
  -e POSTGRES_USER=litellm \
  -e POSTGRES_DB=litellm \
  -p 5544:5432 \
  postgres:16

# Wait for it to accept connections (usually <2 seconds):
until docker exec litellm-pg pg_isready -U litellm 2>/dev/null | grep -q "accepting connections"; do
  sleep 1
done
echo "Postgres ready on 127.0.0.1:5544"
```

### Push the LiteLLM schema

LiteLLM bundles its Prisma schema; we run it once against the fresh DB:

```bash
# Locate LiteLLM's bundled schema
SCHEMA_DIR=$(python -c "import litellm.proxy, os; print(os.path.dirname(litellm.proxy.__file__))")
cd "$SCHEMA_DIR"

# Generate Prisma client (one-time, per-venv)
prisma generate

# Push the schema to your fresh Postgres
DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:5544/litellm \
  prisma db push --accept-data-loss --skip-generate
# → "🚀  Your database is now in sync with your Prisma schema."

cd -
```

If you see `Error: P1010: User 'litellm' was denied access on the
database 'litellm.public'`, your Postgres user needs schema
ownership. Fix:

```bash
docker exec litellm-pg psql -U litellm -d litellm \
  -c "ALTER USER litellm WITH SUPERUSER;"
# Then re-run prisma db push.
```

---

## Step 5 — Write the LiteLLM config

Pick a working directory (e.g. `~/litellm-blockrun-deploy/`) — the
**`custom_callbacks.py` bridge file must live in this same directory**
because LiteLLM Proxy loads callbacks by filename, not by installed
package.

```bash
mkdir -p ~/litellm-blockrun-deploy
cd ~/litellm-blockrun-deploy
```

### `config.yaml`

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: openai/openai/gpt-5.5            # double prefix is intentional:
                                              #   first = LiteLLM provider ("openai-compatible HTTP")
                                              #   second = BlockRun model id
      api_base: http://127.0.0.1:4001/v1      # the BlockRun sidecar
      api_key: "dummy"                        # not used; sidecar handles auth

  - model_name: claude-fable-5
    litellm_params:
      model: openai/anthropic/claude-fable-5
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

  - model_name: deepseek-v4-flash             # FREE — perfect for smoke testing
    litellm_params:
      model: openai/nvidia/deepseek-v4-flash
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

  - model_name: gemini-3.1-pro
    litellm_params:
      model: openai/google/gemini-3.1-pro
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

litellm_settings:
  drop_params: True                            # silently drop OpenAI params BlockRun ignores
  callbacks: ["custom_callbacks.blockrun_logger"]   # JSONL request log

general_settings:
  master_key: "sk-blockrun-demo-master"        # ← change this in prod!
                                               # also becomes the UI password (admin / <master_key>)
```

### `custom_callbacks.py`

```python
# Drop this file next to config.yaml.
# LiteLLM Proxy loads callbacks by relative filename, not by installed package,
# so this 2-line shim is necessary even though blockrun_litellm.logger
# is already on the import path.
from blockrun_litellm.logger import JSONLLogger

blockrun_logger = JSONLLogger()
```

---

## Step 6 — Start LiteLLM Proxy

In **terminal 2** (sidecar must still be running in terminal 1):

```bash
source ~/.venv/litellm-blockrun/bin/activate
cd ~/litellm-blockrun-deploy

export DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:5544/litellm
export BLOCKRUN_LITELLM_LOG=~/litellm-blockrun-deploy/calls.jsonl

litellm --config config.yaml --port 4000 --host 127.0.0.1
```

You should see:

```
   ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
   ...
LiteLLM: Proxy initialized with Config, Set models:
    gpt-5.5
    claude-fable-5
    deepseek-v4-flash
    gemini-3.1-pro
INFO:     Uvicorn running on http://127.0.0.1:4000
```

Verify:

```bash
curl http://127.0.0.1:4000/health/liveliness
# → {"status":"healthy"}

curl -L http://127.0.0.1:4000/ui
# → 200 (HTML page)
```

---

## Step 7 — Open the UI

Browser → **http://127.0.0.1:4000/ui/**

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `sk-blockrun-demo-master` *(whatever you set as `master_key`)* |

> The login page says "Password is your set LiteLLM Proxy MASTER_KEY."
> That's literal — the `master_key` from `config.yaml` is the UI
> password by default. If you want a different UI password, set the
> env var `UI_PASSWORD=...` (the `ui_password` field in `config.yaml`
> is **ignored** — known LiteLLM quirk).

What you can do in the UI:

- **Models** — see the 4 registered models, test calls in the browser
- **Virtual Keys** — generate sub-keys with per-key budgets, model
  allowlists, rate limits. Give one to each customer / team.
- **Logs** — every request that hits the proxy (DB-stored)
- **Spend Analytics** — per-key, per-model, per-day spend
- **Settings** — turn on caching, alerting, etc.

---

## Step 8 — Smoke-test a call

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-blockrun-demo-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"Say one word"}],
    "stream": true
  }'
```

Expected: `text/event-stream` response with a couple of `data: {...}` chunks and a terminating `data: [DONE]`.

**Image generation** (sidecar directly, since v0.3.6):

```bash
curl http://127.0.0.1:4001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "google/nano-banana", "prompt": "a corgi astronaut", "size": "1024x1024"}'
```

Or via LiteLLM:

```python
import litellm
resp = litellm.image_generation(
    model="openai/google/nano-banana",
    prompt="a corgi astronaut",
    api_base="http://127.0.0.1:4001/v1",
    api_key="dummy",
)
print(resp.data[0].url)
```

Then check the local JSONL log appended:

```bash
tail -1 ~/litellm-blockrun-deploy/calls.jsonl | python -m json.tool
```

You'll see a row with `model`, `messages`, `completion`, `usage`,
`latency_ms`, `status`, etc. — see
[`CUSTOMER-ONBOARDING.md`](CUSTOMER-ONBOARDING.md#local-request-log-inputoutput-tokens-latency-cost)
for the full schema.

---

## All persistent files in one table

| File / env var | What | Configurable |
|---|---|---|
| `$BLOCKRUN_WALLET_KEY` | Base private key | yes |
| `$SOLANA_WALLET_KEY` | Solana private key | yes |
| `~/.blockrun/.session` | Auto-created Base wallet | — |
| `~/.blockrun/.solana-session` | Auto-created Solana wallet | — |
| `~/.blockrun/cost_log.jsonl` | Per-paid-call USDC cost (SDK) | — |
| `~/.blockrun/data/*.json` | Full archived request/response for paid calls | — |
| `~/litellm-blockrun-deploy/calls.jsonl` | LiteLLM request log (this adapter) | `BLOCKRUN_LITELLM_LOG` env |
| Docker volume `litellm-pg` data dir | LiteLLM keys / spend / logs (Postgres) | `DATABASE_URL` env |
| `$DATABASE_URL` | LiteLLM Proxy DB connection string | yes |
| `$BLOCKRUN_PROXY_TOKEN` | Optional Bearer guard on the sidecar | yes |

---

## Stopping / restarting

```bash
# Stop everything (in the right order)
kill %1 2>/dev/null   # in the terminal running `litellm`
kill %1 2>/dev/null   # in the terminal running `blockrun-litellm-proxy`
docker stop litellm-pg

# Resume later
docker start litellm-pg                              # Postgres keeps its data
blockrun-litellm-proxy --port 4001 &                 # sidecar
DATABASE_URL=... litellm --config config.yaml --port 4000   # proxy
```

To wipe everything:

```bash
docker rm -f litellm-pg          # destroys the database
rm -rf ~/.blockrun/                # WARNING: deletes your wallet — back it up first!
```

---

## Troubleshooting — the actual gotchas we hit

### `Authentication Error, Not connected to DB!` on UI login

You forgot to set `DATABASE_URL`, or Postgres isn't reachable. Confirm:

```bash
PGPASSWORD=litellm psql -h 127.0.0.1 -p 5544 -U litellm -d litellm -c "SELECT 1"
```

If that fails, your Postgres container isn't running OR you hit the
port-conflict bug below.

### Port 5432 conflict — host has another Postgres

`docker run -p 5432:5432 postgres` will appear to succeed, but
host-side `psql -h 127.0.0.1` connects to the *host's* Postgres, not
the container. Symptom: `role "litellm" does not exist` even though
you just created it inside the container.

Fix: use a free port like 5544 (we do above).

### `Unable to find Prisma binaries. Please run 'prisma generate' first.`

You skipped step 4's `prisma generate`. Run it from LiteLLM's bundled
schema directory:

```bash
SCHEMA_DIR=$(python -c "import litellm.proxy, os; print(os.path.dirname(litellm.proxy.__file__))")
cd "$SCHEMA_DIR" && prisma generate
```

### `ModuleNotFoundError: No module named 'prisma'`

`litellm[proxy]` doesn't include the `prisma` Python package. Add
it explicitly: `pip install prisma`.

### `P1010: User 'litellm' was denied access on the database 'litellm.public'`

Postgres 15+ changed default schema permissions. Grant your user
superuser inside the container:

```bash
docker exec litellm-pg psql -U litellm -d litellm \
  -c "ALTER USER litellm WITH SUPERUSER;"
```

### `Invalid credentials used to access UI`

You set `ui_password` in `config.yaml` and expected it to take effect
— LiteLLM ignores that field. Either:

- Log in with the **`master_key`** as the password (default behavior), or
- Restart with `UI_USERNAME=admin UI_PASSWORD=yours litellm --config ...` (env vars)

### `Could not import proxy_logger from blockrun_litellm.logger`

LiteLLM Proxy loads callbacks **by filename relative to `config.yaml`**,
not by installed-package import. The bridge file
`custom_callbacks.py` must be in the same directory as `config.yaml`,
and you reference it as `"custom_callbacks.blockrun_logger"` — not
`"blockrun_litellm.logger.proxy_logger"`.

### Free model upstream is slow or 503s

The free NVIDIA NIM models (`nvidia/deepseek-v4-flash` etc.) have
flaky upstream availability — 503s and 60-90s response times happen.
The SDK transparently retries 5xx 3× with exponential backoff, but
sometimes the upstream is just down. For production use a paid model
or set up router fallbacks in `config.yaml`:

```yaml
router_settings:
  num_retries: 2
  fallbacks:
    - {"deepseek-v4-flash": ["gpt-5.5"]}
```

### `BLOCKRUN_LITELLM_LOG` is set but no JSONL appears

Check the file path you set is writable by the proxy user, and that
the proxy actually started with that env var visible:

```bash
# Inside the running proxy's terminal, the env should be there
env | grep BLOCKRUN_LITELLM_LOG
```

If you used `~`, expand it (`/Users/you/...`) or use `$HOME` — depending
on your shell, `~` in env vars may not expand.

---

## Production hardening checklist

Before you point real customers at this:

- [ ] Replace `master_key: "sk-blockrun-demo-master"` with a strong
      random value (`openssl rand -hex 32`)
- [ ] Set `BLOCKRUN_PROXY_TOKEN` on the sidecar; clients must send
      `Authorization: Bearer $TOKEN`
- [ ] Run the sidecar bound to `127.0.0.1` only (the default). The
      LiteLLM Proxy is the only thing that should reach it.
- [ ] Set `UI_USERNAME` / `UI_PASSWORD` env vars (don't rely on the
      master_key as UI password)
- [ ] Move Postgres to managed (Neon / Supabase / RDS) — don't rely
      on a single Docker container without a backup strategy
- [ ] Add a router fallback list so a flaky free model doesn't break
      production traffic
- [ ] Set per-virtual-key budgets in the UI so a runaway customer
      can't drain your wallet
- [ ] Wallet hygiene: rotate the BlockRun wallet key periodically,
      keep most USDC in cold storage and top up the hot wallet from
      it
- [ ] Run the whole stack inside a private network if you're not
      using TLS termination upstream

---

## Quick reference — service ports + URLs

| Service | URL | What |
|---|---|---|
| BlockRun sidecar healthz | http://127.0.0.1:4001/healthz | should return `{"status":"ok"}` |
| BlockRun sidecar chat | http://127.0.0.1:4001/v1/chat/completions | chat completions; signs x402 |
| BlockRun sidecar images | http://127.0.0.1:4001/v1/images/generations | image generation (since 0.3.6) |
| BlockRun sidecar models | http://127.0.0.1:4001/v1/models | chat model catalog |
| LiteLLM Proxy health | http://127.0.0.1:4000/health/liveliness | `{"status":"healthy"}` |
| LiteLLM Proxy API | http://127.0.0.1:4000/v1 | what your apps hit |
| LiteLLM Proxy UI | http://127.0.0.1:4000/ui/ | admin dashboard |
| LiteLLM Proxy `/docs` | http://127.0.0.1:4000/docs | Swagger UI |
| Postgres | postgresql://litellm:litellm@127.0.0.1:5544/litellm | DB connection |

---

## Links

- LiteLLM docs: https://docs.litellm.ai
- BlockRun: https://blockrun.ai
- blockrun-litellm GitHub: https://github.com/BlockRunAI/blockrun-litellm
- blockrun-llm SDK: https://github.com/BlockRunAI/blockrun-llm
- x402 protocol: https://x402.org

Questions / bugs: https://github.com/BlockRunAI/blockrun-litellm/issues
