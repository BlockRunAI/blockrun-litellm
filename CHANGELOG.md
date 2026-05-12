# Changelog

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
