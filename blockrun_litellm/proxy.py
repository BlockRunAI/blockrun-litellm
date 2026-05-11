"""
Local OpenAI-compatible proxy for BlockRun.

Run as a sidecar; point any OpenAI client (LiteLLM, langchain, raw SDK,
curl) at ``http://localhost:4001/v1`` and it Just Works. Your x402 wallet
key lives in this process — clients on the same host never see it.

Endpoints
---------
- ``POST /v1/chat/completions`` — OpenAI Chat Completions
- ``GET  /v1/models``           — passthrough to BlockRun's catalog
- ``GET  /healthz``             — liveness probe (no upstream call)

Auth
----
The proxy itself is **unauthenticated by default** — bind to ``127.0.0.1``
in production. Optional shared-secret guard via ``BLOCKRUN_PROXY_TOKEN``;
clients must then send ``Authorization: Bearer <token>``.

CLI
---
::

    $ pip install 'blockrun-litellm[proxy]'
    $ export BLOCKRUN_WALLET_KEY=0x...
    $ blockrun-litellm-proxy --port 4001
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "blockrun-litellm[proxy] extras not installed. "
        "Run: pip install 'blockrun-litellm[proxy]'"
    ) from exc

from blockrun_llm.types import APIError, PaymentError

from blockrun_litellm import _adapter

log = logging.getLogger("blockrun_litellm.proxy")

app = FastAPI(
    title="blockrun-litellm proxy",
    description="OpenAI-compatible front-end for BlockRun's x402 gateway.",
    version="0.1.0",
    docs_url="/docs",
)


# ---------------------------------------------------------------------------
# Optional shared-secret auth
# ---------------------------------------------------------------------------

def _expected_token() -> Optional[str]:
    return os.environ.get("BLOCKRUN_PROXY_TOKEN") or None


def _require_token(authorization: Optional[str] = Header(default=None)) -> None:
    expected = _expected_token()
    if expected is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(401, "Invalid Bearer token")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(_require_token)])
async def list_models() -> Dict[str, Any]:
    # Reuse the cached async client; ``list_models`` does not require payment.
    client = _adapter.get_async_client()
    models = await client.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.get("id"),
                "object": "model",
                "owned_by": m.get("provider", "blockrun"),
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_require_token)])
async def chat_completions(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    model = body.get("model")
    messages = body.get("messages")
    if not model or not isinstance(messages, list):
        raise HTTPException(400, "`model` and `messages` are required")

    # Drop control fields before forwarding the rest as OpenAI kwargs.
    openai_kwargs = {k: v for k, v in body.items() if k not in ("model", "messages")}

    try:
        payload = await _adapter.chat_completion_async(
            model=model,
            messages=messages,
            **openai_kwargs,
        )
    except _adapter.StreamingNotSupported as exc:
        raise HTTPException(400, str(exc))
    except PaymentError as exc:
        # Budget exceeded / wallet underfunded — map to 402 so clients can
        # surface a clean payment-required message.
        raise HTTPException(402, str(exc))
    except APIError as exc:
        # Upstream BlockRun error — preserve status code when reasonable.
        status = exc.status_code if 400 <= getattr(exc, "status_code", 0) < 600 else 502
        return JSONResponse(status_code=status, content={"error": str(exc)})

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="blockrun-litellm-proxy",
        description="OpenAI-compatible local proxy for BlockRun (x402 gateway).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4001, help="Bind port (default: 4001)")
    parser.add_argument(
        "--api-url",
        default=None,
        help="Override BlockRun API URL (default: https://blockrun.ai/api)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args()

    if args.api_url:
        os.environ["BLOCKRUN_API_URL"] = args.api_url

    # Fail fast if no wallet — better than waiting for first request.
    try:
        _adapter.get_sync_client()
    except ValueError as exc:
        parser.exit(2, f"\nWallet not configured:\n  {exc}\n")

    import uvicorn

    uvicorn.run(
        "blockrun_litellm.proxy:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
