# Changelog

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
