# blockrun-litellm — docs

Two guides, two languages. Pick by what you need.

## 中文 / Chinese

| 文档 | 适用 |
|---|---|
| [**CUSTOMER-ONBOARDING.zh.md**](CUSTOMER-ONBOARDING.zh.md) | 5 分钟上手 —— Mode A (Python 库) + Mode B (LiteLLM Proxy)，**推荐用 Solana 钱包** |
| [**PROXY-FULL-SETUP.zh.md**](PROXY-FULL-SETUP.zh.md) | 完整部署（含 LiteLLM 管理 UI + Postgres + Docker），含踩坑清单 + 生产 checklist |

## English

| Document | For |
|---|---|
| [**CUSTOMER-ONBOARDING.md**](CUSTOMER-ONBOARDING.md) | 5-minute onboarding — Mode A (Python lib) + Mode B (LiteLLM Proxy), **Solana wallet recommended** |
| [**PROXY-FULL-SETUP.md**](PROXY-FULL-SETUP.md) | Full deploy (LiteLLM Proxy + admin UI + Postgres + Docker), with troubleshooting + production checklist |

---

## TL;DR

```bash
# Install
pip install -U 'blockrun-litellm[proxy,solana]'

# Set up Solana wallet (recommended — cheaper gas, full feature parity)
python -c "from blockrun_llm import setup_agent_solana_wallet; setup_agent_solana_wallet()"

# Call any model through LiteLLM
import litellm
from blockrun_litellm import register
register()

response = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="https://sol.blockrun.ai/api",   # ← decides chain (Solana)
    stream=True,
)
```

Full instructions: pick a doc above.
