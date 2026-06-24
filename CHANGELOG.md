# Changelog

## 0.4.4 — 2026-06-24

### Changed
- **Require `blockrun-llm>=1.4.6`** — guarantees the SDK attaches the real
  per-call x402 charge to `ChatResponse` (`cost_usd` / `settlement`, race-free).
  The exact-cost reporting added in 0.4.3 now uses the authoritative on-chain
  charge rather than the best-effort `_last_call_cost` fallback.

## 0.4.3 — 2026-06-24

### Added
- **Report the real x402 wallet charge instead of LiteLLM's estimate** (#11).
  In-process custom-provider mode surfaces the SDK's real per-call charge
  (`response.cost_usd`) into `_hidden_params["response_cost"]`, so LiteLLM's
  spend / `max_budget` reflect the **actual wallet deduction** (which carries
  the per-call floor + margin) rather than a token×list-price estimate. Also
  exposes `blockrun_cost_usd` / `blockrun_settlement`, and the JSONL log now
  records `cost_usd` (real when known), `cost_source`
  (`blockrun_x402` vs `litellm_estimate`), `estimated_cost_usd`, and
  `settlement`.
  - Non-streaming in-process path only. Needs a `blockrun-llm` that attaches
    `response.cost_usd` for the race-free path; otherwise falls back to a
    best-effort (`_last_call_cost`) value. Proxy-server mode is unchanged —
    see the per-model pricing in 0.4.2.

## 0.4.2 — 2026-06-24

### Changed
- **Require `blockrun-llm>=1.4.1`** (#8) so `pip install -U blockrun-litellm`
  pulls the SDK fix.

### Docs / examples
- **Example LiteLLM proxy config now sets per-model pricing**
  (`input_cost_per_token` / `output_cost_per_token`, plus cache costs for Claude)
  so team/key budgets (e.g. a `$200` cap) actually enforce — without these the
  proxy prices blockrun model ids at `$0` and the cap never trips. Documents the
  requirement, the unpriced `blockrun/*` wildcard hole, and the cache/streaming
  accuracy caveats; points at the gateway reconciliation report as the exact
  cost source of truth (#10).
- README links back to blockrun.ai/docs (#9).

## 0.4.1 — 2026-06-14

### Fixed
- **`/v1/chat/completions` is now a verbatim x402-signed passthrough — streamed
  tool calls no longer crash.** The endpoint previously went through the SDK's
  typed `chat_completion_stream`, which rejects streaming tool-call
  argument-fragment frames, falls back to `model_construct` (leaving `choices` as
  raw dicts), then crashes in the archive loop with `'dict' object has no
  attribute 'delta'`. It now rides the same `_forward_passthrough` helper as
  `/v1/messages`, so every OpenAI client — including **Codex with
  `wire_api=chat`** — keeps streamed `tool_calls` intact.
- **A Solana RPC fault during x402 signing maps to `503`, not a bare `500`.**
  `_forward_passthrough` (used by `/v1/chat/completions`, `/v1/messages`, and
  `/v1/messages/count_tokens`) now catches a `SolanaRpcException` raised before
  any upstream status exists and surfaces a clean `503`; non-Solana faults still
  propagate. Restores the behaviour the old typed chat path had.
- **`__version__` was stale at `0.3.14`** while `pyproject.toml` had moved to
  `0.4.0`; the two are back in sync.

### Internal
- Generalized `_forward_anthropic` → `_forward_passthrough(headers)` and removed
  the now-dead `_sse_event_stream` / `_openai_error_event` SDK-streaming helpers.
- The litellm `/v1/messages` end-to-end tool-call test auto-skips when the
  installed litellm validates the custom provider against its `LlmProviders`
  enum (it isn't in that enum), so a plain `pytest` is green by default. Added
  passthrough coverage for `/v1/chat/completions` and the new `503` mapping.

## 0.4.0 — 2026-06-12

### Added
- **Native Anthropic `/v1/messages` passthrough (Base + Solana).** The sidecar
  now exposes `POST /v1/messages` and `POST /v1/messages/count_tokens` and
  forwards them verbatim to BlockRun's native Anthropic endpoint, adding only the
  x402 signature — `tools` / `tool_choice` / `thinking` / streaming pass through
  untouched. One sidecar now serves Claude Code (`/v1/messages`), Codex, and any
  OpenAI client with no lossy Anthropic↔OpenAI translation layer. Solana signing
  goes through a new lock-guarded `_SolanaX402Transport`.
- **Streaming tool calls reach the Anthropic bridge with their arguments.** The
  in-process custom provider (`litellm.completion(model="blockrun/…")`) now splits
  each complete SDK tool call into the name/arguments frames LiteLLM's Anthropic
  adapter expects (`_iter_stream_chunks`).

### Fixed
- **Parallel tool calls no longer collapse onto one content block.** BlockRun's
  `ToolCall` carries no per-call `index`, so the previous `getattr(tc, "index", 0)`
  was always `0` — every parallel tool call landed on Anthropic block index 0 and
  all but the last were dropped (the agentic pattern Claude Code uses most). The
  stream now assigns a stream-scoped monotonic block index, fixed across chunks.
- **Streaming upstream errors keep their real status.** A 4xx/5xx on a streaming
  `/v1/messages` was delivered as HTTP 200 `text/event-stream` with a non-SSE
  error body, which the Anthropic SDK would mis-parse or hang on. The upstream is
  now opened before headers are committed, so a genuine error returns a real error
  `Response` with the upstream status and body.
- **The Anthropic routes now respect the concurrency cap.** `/v1/messages` and
  `/v1/messages/count_tokens` were the only paid routes not gated by
  `_get_semaphore()`; under agentic load they could stampede the gateway past
  `BLOCKRUN_MAX_CONCURRENT`. Both now hold the semaphore for the paid upstream
  call (released before the streaming body drain), and `count_tokens` now forwards
  the inbound query string like `/v1/messages`.

## 0.3.14 — 2026-06-11

### Fixed
- **Slow Solana image models no longer time out mid-generation.** The image
  `SolanaLLMClient` is now constructed with an explicit `image_timeout` (default
  300s, overridable via `BLOCKRUN_SOLANA_IMAGE_TIMEOUT`). `openai/gpt-image-2`
  can take well past the SDK's 200s `image_timeout` default on the synchronous
  Solana path; under the default the sidecar threw `httpx.ReadTimeout` before
  the gateway returned the image, surfacing as a 500 (and LiteLLM-Proxy retries)
  even though the gateway had produced the result.
  - This supersedes the original community PR (#1, thanks @KillerQueen-Z), which
    set the general `timeout=` kwarg. Against the current `blockrun-llm` SDK that
    is the **chat** baseline and is overridden per-request for images
    (`_request_image_with_payment` applies `image_timeout`), so it never reached
    image calls. The dedicated `image_timeout=` is the knob that governs them.
  - `tests/test_adapter_solana.py` asserts `image_timeout` is passed (≥ the slow
    tail) and that the env override applies (read at call time, no module reload).

## 0.3.13 — 2026-06-11

### Fixed
- **Streaming `reasoning_content` is now forwarded instead of dropped.**
  Thinking-enabled Claude (Bedrock / direct Anthropic) and native reasoning
  models (DeepSeek R1, GLM thinking) emit their chain-of-thought on the streamed
  delta (`choices[0].delta.reasoning_content`). LiteLLM's
  `GenericStreamingChunk` has no reasoning field and the custom-provider stream
  handler never promotes it onto `delta.reasoning_content`, so
  `_to_generic_chunk` previously lost it on every streamed call. It now routes
  the delta's reasoning into `provider_specific_fields` — the only channel a
  CustomLLM provider has to a live streaming consumer, readable at
  `delta.provider_specific_fields["reasoning_content"]`. Non-stream reasoning is
  unchanged (it already survives verbatim on `message.reasoning_content`). New
  `tests/test_reasoning.py` covers the unit mapping and an end-to-end pass
  through the real `CustomStreamWrapper`.
  - Pairs with the gateway change (blockrun-sol) that forwards extended
    `thinking` to the Bedrock path and emits `reasoning_content` deltas, so a
    Bedrock-served Claude stream now carries reasoning end-to-end.
  - Caveat (LiteLLM limitation): `stream_chunk_builder` does not preserve
    per-chunk `provider_specific_fields`, so a reassembled message will not carry
    `reasoning_content`. Read it from the live delta, or use non-stream for the
    assembled field.

## 0.3.12 — 2026-06-10

### Fixed
- **Provider-mode streaming now forwards the gateway's end-of-stream `usage`
  frame.** The gateway (paired change: blockrun-sol #20 / blockrun #137) emits
  the OpenAI `include_usage` final frame (`choices: [] + usage`) carrying the
  real upstream token counts. `_to_generic_chunk` previously returned
  `usage=None` on the choice-less branch, so LiteLLM never saw those counts and
  re-estimated the prompt with its own tokenizer (~37% drift vs the real
  Anthropic count). The choice-less branch now forwards `chunk.usage` when
  present; older gateways that never send the frame hit the `usage=None` path
  unchanged.
- **The forwarding contract is locked in by `tests/test_stream_usage.py`.**
  The usage frame arrives *after* the finish-reason chunk, and LiteLLM's
  `CustomStreamWrapper` drops any post-finish chunk whose dict lacks the
  `provider_specific_fields` key — the frame only survives because
  `_to_generic_chunk` always emits that key (added in 0.3.10's fingerprint
  passthrough). New tests cover the token counts (unit + end-to-end through
  the real `CustomStreamWrapper` / `stream_chunk_builder`), the key-presence
  contract, and a canary that fails if LiteLLM's post-finish guard ever
  changes.

## 0.3.11 — 2026-06-01

### Added
- **OpenAI Responses API bridge — `POST /v1/responses`.** The sidecar previously
  404'd on `/v1/responses` (only Chat Completions was implemented), so LiteLLM
  clients calling the Responses API got `NotFoundError`. It now accepts a
  Responses request (`input` as string or item list, `instructions`,
  `temperature`, `top_p`, `max_output_tokens`, `tools`), translates it to a chat
  completion against the gateway, and translates the result back: a `response`
  object (with `output[]`, `output_text`, `usage.input/output/total_tokens`) for
  non-streaming, or the canonical `response.*` SSE event sequence
  (`response.created` → `output_item.added` → `content_part.added` →
  `output_text.delta*` → `*.done` → `response.completed`) when `stream=True`.
  Text-in/text-out is fully bridged; Responses-only state (tools-as-state,
  reasoning items, `previous_response_id`, `store`) is not round-tripped — use
  `/v1/chat/completions` for those. New `tests/test_responses.py`.

## 0.3.10 — 2026-06-01

### Added
- **Native fingerprint passthrough is now guaranteed and tested.** The gateway
  returns the upstream provider's response verbatim, so relay-detection signals
  (`system_fingerprint`, `service_tier`, `usage.prompt_tokens_details` /
  `cache_read_input_tokens` / `cache_creation_input_tokens`, per-message
  `reasoning_content`) survive both integration modes. New
  `tests/test_fingerprint.py` locks the contract so a future LiteLLM / SDK bump
  can't silently strip them. New README section "Native fingerprint passthrough".
- **Provider-mode streaming now carries the native fingerprint.** The lossy
  `GenericStreamingChunk` previously dropped everything but text/usage; chunk
  extras (`system_fingerprint`, `service_tier`) are now surfaced via
  `provider_specific_fields` (new `_native_extras()` helper in `provider.py`).

### Changed
- **`blockrun-llm` floor raised to `>=0.37.0`** (runtime + `[solana]` extra) for
  concurrency-safe Solana payments. The adapter shares one cached SDK client
  across the proxy's concurrent requests; with the older floor that shared client
  raced on x402 nonce/auth state under load and returned a small fraction of
  `Payment verification failed` / `authorization already used` rejections. 0.37.0
  adds a per-client signing lock + whole-request payment retry, taking concurrent
  single-wallet load to ~100% (verified opus-4.7 / gemini-3.1-pro / gpt-5.5
  100/100 at concurrency 10). No proxy code change needed — the bump is enough.
- **Model ids aligned to the current gateway flagships** across the example
  config, README, and example scripts: `anthropic/claude-opus-4-5` →
  `anthropic/claude-opus-4-7`, `google/gemini-3-pro` → `google/gemini-3.1-pro`.
  `openai/gpt-5.5` unchanged.

## 0.3.8 — 2026-05-27

### Fixed
- **Solana image generation now uses `SolanaLLMClient`.** `get_image_client()`
  previously returned the EVM-only `ImageClient` regardless of `api_url`, so
  Solana image requests signed EIP-712 payments and crashed at x402 settlement
  with `transaction_simulation_failed`. Branches on `_is_solana_url(api_url)`
  the same way `get_sync_client()` / `get_async_client()` already did.
- New `_invoke_image_generate()` dispatcher calls `SolanaLLMClient.image()`
  on Solana and `ImageClient.generate()` on Base from both sync and async
  paths.

### Changed
- `blockrun-llm` floor raised to `>=0.31.1` (sync + solana extras) for the
  new `SolanaLLMClient.image()` method.

## 0.3.9 — 2026-05-28

### Fixed
- **Sidecar now surfaces the gateway's real 402 reason.** Previously a
  `PaymentError` from the SDK was returned as a flat
  `{"detail": "Payment rejected. Check your Solana USDC balance."}`,
  even when the real failure was an x402 facilitator settlement error
  (`transaction_simulation_failed`, `insufficient_funds`,
  `payment_expired`, etc.). With `blockrun-llm >= 0.32.0` the SDK
  preserves the gateway body in `PaymentError.response`, and the proxy
  now returns it as `{"error": "...", "details": "..."}` on both the
  `/v1/chat/completions` and `/v1/images/generations` routes, plus
  folds it into the streaming error event so SSE clients see it too.
- **Slow image models (`openai/gpt-image-2`, `openai/dall-e-3`,
  `google/nano-banana-pro` at 4K) now complete via the SDK's new
  202 + poll loop instead of crashing with a Pydantic ValidationError
  on the job stub.** No proxy code changed for this — bumping the
  `blockrun-llm` floor to `>=0.32.0` is enough.

### Changed
- **`blockrun-llm` floor raised to `>=0.32.0`** (for both the runtime
  install and the `[solana]` extra) so the PaymentError surfacing and
  image-poll fix are guaranteed to be present.

## 0.3.7 — 2026-05-19

### Changed

- **Default `BLOCKRUN_MAX_CONCURRENT` raised from 20 → 100.**
  The previous default was a conservative development-time value. The httpx
  pool is configured for 200 connections; each paid request uses 2 connections
  (402 probe + authenticated call), so 100 concurrent requests is the natural
  ceiling before the pool itself becomes the bottleneck.

- **Streaming semaphore released after first chunk (early release).**
  Previously the semaphore was held for the entire stream duration, meaning a
  slow client reading at 1 token/s occupied a concurrency slot for the full
  generation time (potentially 30–60 s). Now the semaphore is released as soon
  as the first chunk arrives from upstream — covering only the x402 probe +
  sign + HTTP handshake. Remaining chunks stream freely without holding a slot.
  In practice this means 100 concurrent streams can proceed simultaneously even
  with very slow clients; the upstream provider's RPM/TPM is the actual limit.

- **Image generation uses a bounded thread pool (max 20 workers).**
  Previously `image_generation_async` used `loop.run_in_executor(None, ...)`,
  which dispatches to Python's default unbounded thread pool. Under heavy image
  load this could spawn hundreds of threads. Now uses a module-level
  `ThreadPoolExecutor(max_workers=20)` that caps image concurrency and prevents
  memory exhaustion.

## 0.3.6 — 2026-05-18

### Added

- **`POST /v1/images/generations` endpoint in the sidecar proxy.**
  The proxy now exposes an OpenAI-compatible image generation endpoint.
  Accepts `prompt`, `model`, `size`, `n`; returns OpenAI `ImageResponse`
  format (`created` + `data[].url`). Supports any model in BlockRun's
  image catalog: `google/nano-banana`, `google/nano-banana-pro`,
  `openai/dall-e-3`, `openai/gpt-image-1`, `openai/gpt-image-2`,
  `xai/grok-imagine-image`, `zai/cogview-4`, etc.
  Uses `ImageClient` under the hood (sync, run via thread executor for
  async compatibility). Cached per wallet/url like all other clients.

## 0.3.5 — 2026-05-15

### Fixed

- **Sidecar proxy: `MidStreamFallbackError` on paid models under concurrent load.**
  Two root causes, both fixed:

  1. **Wrong SSE error format.** When BlockRun's upstream returned a 5xx
     mid-stream, the sidecar embedded the error as
     `data: {"error": {"type": ..., "status": ...}}`.  The `openai` Python SDK
     (v1.x, Rust parser) only recognises error events that follow OpenAI's
     exact schema: `{"error": {"message", "type", "param", "code"}}`.  The
     non-conforming `"status"` key caused the Rust untagged-enum parser to fail
     with *"data did not match any variant of untagged enum Resp"*, which
     LiteLLM then wrapped as a confusing `MidStreamFallbackError`.  Fixed by
     adding `_openai_error_event()` that always emits the correct four-field
     schema.

  2. **No concurrency cap.** 100 simultaneous requests to a paid model like
     `claude-opus-4.7` all signed x402 payments and hit the Anthropic upstream
     concurrently, exhausting its TPM/RPM limits and causing 500s.  The sidecar
     now acquires an `asyncio.Semaphore` before each request (streaming and
     non-streaming alike), limiting in-flight requests to `BLOCKRUN_MAX_CONCURRENT`
     (default 20, configurable via env var).  Excess requests queue inside the
     sidecar rather than flooding the upstream.

### Added

- `BLOCKRUN_MAX_CONCURRENT` env var — set on the sidecar to tune the
  concurrency cap.  Default is 20, which keeps Anthropic's paid-tier limits
  comfortable.  Free models (e.g. `nvidia/deepseek-v4-flash`) are not subject
  to Anthropic's caps so you can raise this for free-only workloads.

## 0.3.4 — 2026-05-14

### Changed
- **Dependency bump:** `blockrun-llm>=0.24.0` (`solana` extra:
  `blockrun-llm[solana]>=0.24.0`). The new SDK switches the default
  Solana RPC endpoint to BlockRun's own multi-region Tatum-backed
  proxy (`https://sol.blockrun.ai/api/v1/solana/rpc`) — partners no
  longer need to register their own Helius / Tatum / QuickNode
  account. See `blockrun-llm 0.24.0` changelog for the migration
  details. Zero code change in this adapter for the change to take
  effect; just upgrade and the default URL flows through.

### Verified e2e
- `litellm.completion(model="blockrun/zai/glm-5.1", api_base="https://sol.blockrun.ai/api", api_key=<solana_key>)`
  — paid Solana settlement succeeded, $0.001 USDC debited on-chain,
  cost log recorded `network=solana-mainnet` /
  `client=SolanaLLMClient`. Blockhash fetched via the BlockRun
  proxy (no partner-side RPC config needed).

## 0.3.3 — 2026-05-13

### Fixed
- **Critical: sidecar (Mode B) Solana paid calls were silently routing
  through the Base async client.** Caused
  ``EncodingTypeError: Value 'EPjFW...' of type <class 'str'> cannot be
  encoded by AddressEncoder`` when the EVM EIP-712 encoder met the
  Solana USDC mint address (base58, not 0x-hex).

  Root cause: ``_is_solana_url(api_url)`` only checked the function
  argument, never the ``BLOCKRUN_API_URL`` env var. The FastAPI
  sidecar's request handlers don't forward an ``api_url`` arg to the
  adapter — they configure the chain via env var at startup
  (``blockrun-litellm-proxy --api-url https://sol.blockrun.ai/api``).
  So ``_is_solana_url(None)`` always returned ``False`` and routed
  every request to ``AsyncLLMClient`` (Base) regardless of the
  sidecar's actual chain. Mode A wasn't affected because callers pass
  ``api_base`` explicitly.

  Fix: ``_is_solana_url`` now falls back to ``BLOCKRUN_API_URL`` when
  no argument is passed, so sidecar requests reach
  ``AsyncSolanaLLMClient`` correctly.

  Free-model calls escaped detection because they skip the payment
  encoder entirely; only paid Solana calls hit the bug.

## 0.3.2 — 2026-05-12

### Fixed
- **Tool calling works on Solana now.** The adapter previously dropped
  ``tools`` / ``tool_choice`` on the Solana path because the SDK didn't
  accept them. Combined with ``blockrun-llm 0.22.1`` (which adds
  ``tools`` / ``tool_choice`` to the Solana SDK methods), function
  calling now works uniformly on Base **and** Solana. The adapter's
  kwarg filter no longer special-cases the Solana chain — every chain
  forwards the same OpenAI-style params.

### Dependency bump
- ``blockrun-llm>=0.22.1``.

## 0.3.1 — 2026-05-12

### New
- **Async Solana is now supported.** ``litellm.acompletion(...)`` and
  ``litellm.acompletion(stream=True)`` with ``api_base="https://sol.blockrun.ai/api"``
  no longer raise ``NotImplementedError``. The adapter routes async
  Solana calls to the new ``AsyncSolanaLLMClient`` introduced in
  ``blockrun-llm 0.22.0``, completing parity with Base across both
  sync/async × stream/non-stream.

### Improved
- **Paid streaming now writes the local cost log + archive.** This is
  inherited from ``blockrun-llm 0.22.0``: every paid streaming call —
  Base or Solana, sync or async — produces a row in
  ``~/.blockrun/cost_log.jsonl`` and a full request/response archive
  in ``~/.blockrun/data/`` once the stream finishes. Closes the audit
  gap that streaming had relative to non-streaming.

### Dependency bump
- ``blockrun-llm>=0.22.0``.

### Verified e2e
- ``await litellm.acompletion(model="blockrun/nvidia/deepseek-v4-flash",
  api_base="https://sol.blockrun.ai/api", api_key=solana_key,
  stream=True)`` returned ``"Hello! How can I"`` — full async Solana
  stream chain through LiteLLM, first attempt.

## 0.3.0 — 2026-05-12

### New
- **Solana chain support.** The adapter now dispatches to
  ``SolanaLLMClient`` when ``api_url`` / ``api_base`` points at the
  Solana gateway (``sol.blockrun.ai``), and to ``LLMClient`` (Base)
  otherwise. Both modes (Python library + LiteLLM Proxy sidecar)
  understand both chains.

  ```python
  # Base (default)
  litellm.completion(
      model="blockrun/openai/gpt-5.5",
      messages=[...],
      api_key="0xBASE_PRIVATE_KEY",
  )

  # Solana
  litellm.completion(
      model="blockrun/openai/gpt-5.5",
      messages=[...],
      api_base="https://sol.blockrun.ai/api",   # ← decides chain
      api_key="solana-private-key",
  )
  ```

- **Streaming on Solana** end-to-end. Requires
  ``blockrun-llm>=0.21.0`` which adds
  ``SolanaLLMClient.chat_completion_stream``.
- ``tools`` / ``tool_choice`` are silently dropped on the Solana path
  (the Solana SDK doesn't support function calling yet) instead of
  raising — same UX as LiteLLM's ``drop_params``.

### Constraints
- **Solana async is not implemented.** ``litellm.acompletion(...)`` with
  a Solana ``api_base`` raises ``NotImplementedError`` because the SDK
  has no async Solana client. Use sync, or wrap in
  ``asyncio.to_thread()``. Roadmap.
- Solana support requires the optional extra: ``pip install
  'blockrun-litellm[solana]'`` (pulls the x402 SVM toolchain). A clear
  ``ImportError`` fires at first call if it's missing.

### Tests
- 8 new unit tests in ``tests/test_adapter_solana.py`` covering URL
  recognition, the ``tools`` drop on Solana, async-Solana
  ``NotImplementedError``, sync routing to ``SolanaLLMClient``, and a
  clear error when the extra is missing.

### Verified e2e
- Live ``litellm.completion(api_base="https://sol.blockrun.ai/api",
  stream=True)`` returned `"Hello! How can I"` over a real Solana
  wallet (loaded from ``~/.blockrun/.solana-session``).

### Dependency bump
- ``blockrun-llm>=0.21.0`` (Solana streaming).
- New ``[solana]`` extra → ``pip install 'blockrun-litellm[solana]'``.

## 0.2.2 — 2026-05-12

### New
- **Local JSONL request logger.** Opt-in observability that captures
  every LiteLLM call — input messages, completion content, token
  usage (`prompt_tokens` / `completion_tokens` / `total_tokens`),
  latency, `stream` flag, optional `cost_usd`, and on failures the
  `error_type` + `error_message`. Writes one JSON object per line to
  `~/.blockrun/litellm_calls.jsonl` (override via
  `BLOCKRUN_LITELLM_LOG` env var or explicit `path` arg).
- **Two integration paths:**
  - **Python library mode** — one line at startup:
    ```python
    from blockrun_litellm import enable_local_logging
    enable_local_logging()
    ```
  - **LiteLLM Proxy Server mode** — drop a tiny `custom_callbacks.py`
    bridge next to `config.yaml` (LiteLLM Proxy loads callbacks by
    filename, not by installed package), then reference
    `["custom_callbacks.blockrun_logger"]` in `litellm_settings.callbacks`.
    See [`examples/custom_callbacks.py`](examples/custom_callbacks.py).
- Built on LiteLLM's `CustomLogger` interface, so **streaming failures
  are captured uniformly** alongside success and non-stream paths.
  Intermediate streaming-chunk fires are deduplicated — exactly one
  row per call.

### Tests
- 13 new unit tests in `tests/test_logger.py` covering: success/
  failure rows, intermediate-chunk dedupe, dict-vs-pydantic response
  shapes, async hook delegation, env-var path resolution,
  `enable_local_logging` idempotency, and the module-level
  `proxy_logger` singleton.

### Verified e2e
- **Mode A** — `enable_local_logging()` + `litellm.completion()`
  writes one JSONL row per call (live BlockRun + LiteLLM 1.83.14).
- **Mode B** — full chain `LiteLLM Proxy (4000) → BlockRun sidecar
  (4001) → blockrun.ai` with `callbacks: ["custom_callbacks.blockrun_logger"]`
  produces a row with `completion="Serendipity"`,
  `usage={prompt_tokens:10, completion_tokens:4, total_tokens:14}`.

## 0.2.1 — 2026-05-12

### Improved
- **Transient errors now translate to LiteLLM's retriable exception
  hierarchy** so the router's own fallback and retry machinery picks
  up where the SDK leaves off. Mapping (in `_translate_to_litellm`):
  - `APIError(500)` → `litellm.InternalServerError`
  - `APIError(502)` / `APIError(504)` → `litellm.APIConnectionError`
  - `APIError(503)` → `litellm.ServiceUnavailableError`
  - `APIError(429)` → `litellm.RateLimitError`
  - `httpx.TimeoutException` → `litellm.Timeout`
  - `httpx.NetworkError` → `litellm.APIConnectionError`
  Everything else propagates unchanged so callers see the real error.
  Applied uniformly to `completion()`, `acompletion()`, `streaming()`,
  and `astreaming()`. Combined with the SDK's in-band 5xx retry (added
  in `blockrun-llm` 0.20.1), most transient hiccups self-heal before
  ever reaching the caller — and the ones that don't are now marked
  retriable so `litellm.Router` can switch providers automatically.
- **Dependency bump:** `blockrun-llm>=0.20.1` (the SDK release that
  ships the matching retry / `fallback_models` improvements).

### Tests
- Seven new unit tests in `tests/test_provider.py` covering the 5xx /
  429 / Timeout / Network translations and verifying that
  non-transient errors (e.g. `RuntimeError`) pass through unchanged.
- Replaced the stale `test_streaming_request_is_rejected` test
  (which asserted v0.1.0's `StreamingNotSupported`) with a kwarg-leak
  check that the non-streaming entrypoint silently drops `stream=True`.

## 0.2.0 — 2026-05-12

### New
- **Streaming (`stream=True`) end-to-end.** Both integration modes now
  speak SSE:
  - **Provider mode** — `BlockRunLLM` implements
    `CustomLLM.streaming()` and `astreaming()`, yielding LiteLLM
    `GenericStreamingChunk` objects so `litellm.completion(..., stream=True)`
    just works.
  - **Proxy mode** — `POST /v1/chat/completions` returns
    `text/event-stream` when `stream=True`, emitting OpenAI-style
    `data: <json>\n\n` events and a terminating `data: [DONE]`. Errors
    raised mid-stream are surfaced as a final `data: {"error": ...}`
    event rather than HTTP errors (headers already flushed).
- **Adapter API:** `_adapter.chat_completion_stream_sync()` and
  `chat_completion_stream_async()` are the new public entrypoints.
  `StreamingNotSupported` is removed — the previous "fail-fast at the
  adapter" behavior is no longer needed.
- **Dependency bumped:** `blockrun-llm>=0.20.0` (this is the SDK release
  that introduces `chat_completion_stream()`).

### Removed
- `_adapter.StreamingNotSupported` — replaced by real streaming support.

### Notes
- Caveats inherited from the BlockRun gateway: `search_parameters` and
  the Responses-API models (`codex`, `gpt-5.4-pro`) reject streaming
  server-side with 400. The adapter does not pre-filter those — clients
  see the same 400 their own LiteLLM call would surface.

## 0.1.0 — 2026-05-11

Initial release. Published to PyPI as
[`blockrun-litellm`](https://pypi.org/project/blockrun-litellm/0.1.0/) and
GitHub at [BlockRunAI/blockrun-litellm](https://github.com/BlockRunAI/blockrun-litellm).

### Verified
- Provider mode (`litellm.completion(model="blockrun/nvidia/deepseek-v4-flash", ...)`)
  returns a real `litellm.ModelResponse` from the live BlockRun gateway.
- Proxy mode (`POST http://127.0.0.1:4001/v1/chat/completions`) returns a
  valid OpenAI Chat Completions JSON response from the live gateway.
- Fresh-venv install (`pip install 'blockrun-litellm[proxy]'`) imports
  cleanly and registers the `blockrun-litellm-proxy` CLI.

### Added
- `BlockRunLLM` — LiteLLM `CustomLLM` handler routing through the BlockRun gateway.
- `register()` — idempotent registration of the `blockrun` provider in `litellm.custom_provider_map`.
- `blockrun-litellm-proxy` — local OpenAI-compatible FastAPI proxy (`pip install 'blockrun-litellm[proxy]'`).
- Endpoints on the proxy: `POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`, `GET /docs`.
- Optional shared-secret guard via `BLOCKRUN_PROXY_TOKEN`.
- Examples: `python_lib.py`, `raw_openai_sdk.py`, `litellm_config.yaml`.
- Bilingual README (English + 中文).

### Known limitations
- **Streaming (`stream=True`) is not wired in this adapter** — surfaces as HTTP 400. The BlockRun gateway itself fully supports SSE (`text/event-stream`) for both free and paid models; the gap is on the `blockrun-llm` SDK client side, which this adapter wraps. Earlier copy in this file blamed "x402 per-request settlement" — that was wrong, x402 is orthogonal to SSE.
- During e2e the pydantic serializer emits a warning that LiteLLM's `Message` model expects 9 fields and BlockRun returns 5 (extras like `function_call`, `audio`, `annotations` are absent). Functionally harmless — LiteLLM fills defaults — but apps with strict pydantic validation may want to suppress the warning.
- **Solana wallet path not wired** — Base only for v0.1. Solana support will land alongside `SolanaLLMClient` integration.
- Image / video / music generation endpoints are not exposed by the proxy yet; only `/v1/chat/completions`.
