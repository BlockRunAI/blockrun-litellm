# BlockRun × LiteLLM — 5 分钟上手指南

> 验证日期：2026-05-15
> 版本：`litellm 1.83.x`、`blockrun-litellm 0.3.5`、`blockrun-llm 0.24.1`、Python 3.13
> 下面每一行命令都是真实跑过的，输出是实测结果
>
> **两条链 + 两种集成模式：**
>
> | 链 | 网关 URL | 钱包环境变量 | 推荐度 |
> |---|---|---|---|
> | **Solana**（USDC） | `https://sol.blockrun.ai/api` | `SOLANA_WALLET_KEY` | ⭐ **首选** — 实测中位 settlement 106ms（Base 935ms，**快 8.8×**），燃料费 < $0.001/笔 |
> | Base（USDC） | `https://blockrun.ai/api`（默认） | `BLOCKRUN_WALLET_KEY` | 99.9% 成功率（vs Solana 94.7%），适合做 Solana 的 fallback |
>
> | 你的部署 | 选 |
> |---|---|
> | 代码里直接调 `litellm.completion(...)` | **模式 A — Python 库** |
> | 跑 `litellm --config ...` 作中央网关 | **模式 B — LiteLLM Proxy Server** |
>
> 想要带 UI 的完整部署？看 [`PROXY-FULL-SETUP.zh.md`](./PROXY-FULL-SETUP.zh.md)

---

## 决策树

```
你跑 LiteLLM Proxy Server 当中央网关吗？
├─ 是 → 模式 B
└─ 否 → 模式 A

你想用哪条链付费？
├─ Solana USDC（推荐：便宜、快、功能对齐）→ SOLANA_WALLET_KEY 环境变量
│                                              + api_base="https://sol.blockrun.ai/api"
└─ Base USDC                                  → BLOCKRUN_WALLET_KEY 环境变量
```

---

## 安装（两种模式都需要）

```bash
# 只用 Base（最小）
pip install -U 'blockrun-litellm[proxy]'

# Base + Solana（推荐 — 加上 x402 SVM 工具链）
pip install -U 'blockrun-litellm[proxy,solana]'
```

需要 Python ≥ 3.9（我们用 3.13 测的）。

---

## 钱包配置（一次性）

**陛下推荐：用 Solana 钱包。** 燃料费便宜（< $0.001/笔），结算快，和 Base 功能已完全对齐。

### Solana（首选）

```bash
# 方式 A — 已有 Solana base58 私钥
export SOLANA_WALLET_KEY=你的BASE58_SOLANA_私钥

# 方式 B — 自动生成 + 扫码充值 USDC（Solana 主网）
python -c "from blockrun_llm import setup_agent_solana_wallet; setup_agent_solana_wallet()"
# 私钥保存到 ~/.blockrun/.solana-session，下次自动加载
```

### Base（备选）

```bash
export BLOCKRUN_WALLET_KEY=0x你的Base私钥
# 或者自动生成：
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"
# 保存到 ~/.blockrun/.session
```

> 🔒 私钥**永不离开你的机器** —— 只用于本地签 EIP-712（Base）或 Ed25519（Solana）支付凭证，签名才上链。和 MetaMask / Phantom 钱包同样的安全模型。

> 💡 想不花 USDC 验证一下？用免费模型 `nvidia/deepseek-v4-flash` —— 同样的代码路径，零结算，Solana 和 Base 都支持。

---

## 模式 A — Python 库

代码里已经在调 `litellm.completion()` 就用这个。

### Solana 用法（推荐）

```python
import litellm
from blockrun_litellm import register
register()  # 启动调一次

# 非流式
response = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=128,
    api_base="https://sol.blockrun.ai/api",   # ← 这一行决定走 Solana
    # api_key=...                              # 可选；省略则用 SOLANA_WALLET_KEY 环境变量
)
print(response.choices[0].message.content)

# 流式
for chunk in litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    api_base="https://sol.blockrun.ai/api",
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Base 用法

```python
import litellm
from blockrun_litellm import register
register()

# 不指定 api_base 默认就是 Base
response = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "你好"}],
)
```

### 异步 + 流式（Solana 和 Base 都支持，0.3.1 起）

```python
import asyncio
import litellm
from blockrun_litellm import register
register()

async def main():
    async for chunk in await litellm.acompletion(
        model="blockrun/openai/gpt-5.5",
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
        api_base="https://sol.blockrun.ai/api",
    ):
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(main())
```

**实测输出（Solana，2026-05-12）：** `"Hello! How can I"` —— 2 个 chunk、约 1 秒。

### 工具调用 / Function Calling

Base 和 Solana 都支持，写法一样：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = litellm.completion(
    model="blockrun/openai/gpt-5.5",
    messages=[{"role": "user", "content": "查询东京天气"}],
    tools=tools,
    tool_choice="auto",
    api_base="https://sol.blockrun.ai/api",   # Solana 也支持
)
# response.choices[0].message.tool_calls 里能看到调用参数
```

---

## 模式 B — LiteLLM Proxy Server

跑 `litellm --config config.yaml` 当中央网关用就走这个。

架构：

```
┌────────────┐    ┌────────────┐     ┌─────────────────┐     ┌──────────────────────────────┐
│ 你的应用   │───▶│ LiteLLM    │────▶│ blockrun-       │────▶│ blockrun.ai (Base)            │
│ (任意语言)│    │ Proxy      │     │ litellm-proxy   │     │ 或 sol.blockrun.ai (Solana)   │
│            │    │ (端口 4000)│     │ (端口 4001)     │     │                              │
└────────────┘    └────────────┘     └─────────────────┘     └──────────────────────────────┘
                                              │
                                              │ 本地签名
                                              ▼
                                       ~/.blockrun/.session       ← Base 钱包
                                       ~/.blockrun/.solana-session ← Solana 钱包
```

### 1. 启动 BlockRun sidecar（终端 1）

```bash
# Solana 链（推荐）
export SOLANA_WALLET_KEY=你的Solana私钥
blockrun-litellm-proxy --port 4001 --api-url https://sol.blockrun.ai/api

# 或 Base 链
# export BLOCKRUN_WALLET_KEY=0x你的Base私钥
# blockrun-litellm-proxy --port 4001
```

sidecar：
- 在 `:4001/v1` 提供 OpenAI 兼容 HTTP 接口
- 底层处理 x402 签名
- 默认绑 `127.0.0.1`（loopback，安全）

### 2. `config.yaml` 加上 BlockRun 模型

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: openai/openai/gpt-5.5            # 双 'openai/' 是故意的：
                                              #   前者 = LiteLLM provider 前缀
                                              #   后者 = BlockRun 模型 id
      api_base: http://127.0.0.1:4001/v1      # ← 指向 sidecar
      api_key: "dummy"                        # 不用；sidecar 负责认证

  - model_name: deepseek-v4-flash
    litellm_params:
      model: openai/nvidia/deepseek-v4-flash
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

  - model_name: claude-opus-4-7
    litellm_params:
      model: openai/anthropic/claude-opus-4-7
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

litellm_settings:
  drop_params: True   # 静默丢弃 BlockRun 不支持的 OpenAI 参数
                      # （frequency_penalty / presence_penalty / logit_bias / n）
```

### 3. 启动 LiteLLM Proxy（终端 2）

```bash
litellm --config config.yaml --port 4000 --host 127.0.0.1
```

### 4. 像 OpenAI 接口一样调用

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-你的-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"说一个词"}],
    "stream": true
  }'
```

**实测输出（2026-05-12）：** HTTP 200、`text/event-stream`、内容 `"Okay."`。

---

## 本地请求日志（含 input/output tokens、延迟、cost）

`blockrun-litellm 0.2.2+` 自带一行启用的 JSONL 日志。

### 日志文件位置

| 来源 | 路径 |
|---|---|
| 显式传给 `enable_local_logging("...")` | 你传的路径 |
| `BLOCKRUN_LITELLM_LOG` 环境变量 | 它指向的路径 |
| 都没设置 | **`~/.blockrun/litellm_calls.jsonl`**（默认） |

目录不存在会自动创建，文件永远追加（不自动轮转 —— 大了用 `logrotate` 或定期 truncate）。一行一个 JSON。

### 每行包含

```
ts, iso, model, provider, messages, completion,
usage{prompt_tokens, completion_tokens, total_tokens},
latency_ms, stream, cost_usd, status,
error_type, error_message, request_id
```

### 模式 A 启用（一行）

```python
from blockrun_litellm import enable_local_logging
enable_local_logging()                   # 默认 ~/.blockrun/litellm_calls.jsonl
# 或 enable_local_logging("/var/log/calls.jsonl")
```

### 模式 B 启用（加一个桥接文件）

LiteLLM Proxy 按**文件名加载** callback，不读已安装的包，所以在 `config.yaml` 同目录加一个 `custom_callbacks.py`：

```python
# custom_callbacks.py
from blockrun_litellm.logger import JSONLLogger
blockrun_logger = JSONLLogger()
```

`config.yaml` 里引用：

```yaml
litellm_settings:
  callbacks: ["custom_callbacks.blockrun_logger"]
```

两种模式都识别 `BLOCKRUN_LITELLM_LOG` 环境变量。

### 实测一行（真实记录）

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

### 聚合分析

纯 JSONL，`jq` / `pandas` / `duckdb` 都能直接吃：

```bash
# 过去一小时调了多少次，总成本
jq -s 'map(select(.ts > (now - 3600))) | {n: length, cost: (map(.cost_usd // 0) | add)}' \
   ~/.blockrun/litellm_calls.jsonl
```

```python
import json
from collections import Counter
rows = [json.loads(l) for l in open("~/.blockrun/litellm_calls.jsonl")]
by_model = Counter(r["model"] for r in rows)
print(by_model.most_common())
```

---

## 所有持久化文件一览表

| 文件 / 环境变量 | 内容 | 可改吗 |
|---|---|---|
| `BLOCKRUN_WALLET_KEY`（env） | Base 私钥 | 是 |
| `SOLANA_WALLET_KEY`（env） | Solana 私钥 | 是 |
| `~/.blockrun/.session` | 自动生成的 Base 钱包 | — |
| `~/.blockrun/.solana-session` | 自动生成的 Solana 钱包 | — |
| `~/.blockrun/litellm_calls.jsonl` | LiteLLM 请求日志（本适配层） | `BLOCKRUN_LITELLM_LOG` env 或 `enable_local_logging(path)` |
| `~/.blockrun/cost_log.jsonl` | 付费 USDC 成本审计（SDK 写） | 不可改 |
| `~/.blockrun/data/*.json` | 付费调用的完整请求/响应归档（SDK 写） | 不可改 |
| `BLOCKRUN_PROXY_TOKEN`（env） | sidecar 可选的共享密钥 | 是 |

---

## 错误处理

`blockrun-litellm` 把上游临时错误翻译成 LiteLLM 可重试的异常类型，`litellm.Router` 自动处理：

| BlockRun 上游错误 | 你的应用看到 |
|---|---|
| 503 Service Unavailable | `litellm.ServiceUnavailableError` |
| 502 Bad Gateway / 504 Gateway Timeout | `litellm.APIConnectionError` |
| 500 Internal Server Error | `litellm.InternalServerError` |
| 429 Too Many Requests | `litellm.RateLimitError` |
| 读取/连接超时 | `litellm.Timeout` |

LiteLLM router 配置直接生效：

```yaml
router_settings:
  num_retries: 2
  fallbacks:
    - {"gpt-5.5": ["claude-opus-4-7"]}   # gpt-5.5 出 503 自动切 claude
```

SDK 自身也会做 5xx 退避重试 3 次（1s / 2s / 4s），大部分临时抖动都自愈了。

要让 SDK 在第一个 chunk 之前先走 fallback 模型链：

```python
litellm.completion(
    model="blockrun/nvidia/deepseek-v4-flash",
    messages=[...],
    fallback_models=["nvidia/llama-4-maverick"],   # BlockRun SDK 接管
    api_base="https://sol.blockrun.ai/api",
)
```

---

## 流式

✅ 完整支持（自 `blockrun-litellm 0.2.0` Base、`0.3.0` Solana）。SSE chunks 原样透传（`text/event-stream`，`data: <json>\n\n` + `data: [DONE]`）。

**网关层硬限制**（服务端返 HTTP 400）：

- OpenAI Responses API 系列模型（`codex`、`gpt-5.4-pro`）不支持流式
- xAI Live Search（`search_parameters`）不能流式

---

## 常见问题

### Q：还需要 OpenAI / Anthropic / Google 的 API key 吗？

不需要。BlockRun 是统一网关，一个钱包付所有家的费。原有的 vendor-specific key 这里用不上。

### Q：每次调用要花多少钱？

- **免费 tier**：`nvidia/deepseek-v4-flash`、`nvidia/llama-4-maverick` 等 $0
- **付费**：按 token 计价，每个模型不同；调 `/v1/models` 看实时目录，或参考 [blockrun.ai](https://blockrun.ai)。典型 Claude Opus 4 一次大约 $0.001–0.01

每笔付费成本记录在 `~/.blockrun/cost_log.jsonl`：

```python
from blockrun_llm import get_cost_log_summary
print(get_cost_log_summary(group_by="model"))
```

### Q：钱包 USDC 不够了会怎样？

网关返 `402 Payment Required`。配了 router fallback 的话，LiteLLM 自动切到下一个 provider；没配就抛 `litellm.AuthenticationError`，你充值后再调。

### Q：怎么在 Docker / k8s 里跑？

标准 FastAPI 应用：

```dockerfile
FROM python:3.13-slim
RUN pip install 'blockrun-litellm[proxy,solana]'
# 通过 secret manager 注入私钥 —— 永远别打进镜像层：
ENV SOLANA_WALLET_KEY=""
ENV BLOCKRUN_WALLET_KEY=""
CMD ["blockrun-litellm-proxy", "--host", "0.0.0.0", "--port", "4001", "--api-url", "https://sol.blockrun.ai/api"]
```

多租户部署设 `BLOCKRUN_PROXY_TOKEN=...`，客户端用 `Authorization: Bearer $TOKEN` 调用。

### Q：怎么看有哪些模型可用？

```bash
curl http://127.0.0.1:4001/v1/models                  # 通过 sidecar
curl https://sol.blockrun.ai/api/v1/models            # Solana 直接
curl https://blockrun.ai/api/v1/models                # Base 直接
```

### Q：选 Solana 还是 Base？

**强烈推荐 Solana。** 基于最近 3 天（2026-05-10 ~ 05-12）生产环境 4,424 条实际结算记录：

| 指标 | **Solana** (PayAI facilitator) | Base (Coinbase CDP) | Solana 优势 |
|---|---|---|---|
| **中位 settlement 时间** | **106 ms** ⚡ | 935 ms | **8.8× 更快** |
| p90 settlement | 133 ms | 1,658 ms | 12.5× 更快 |
| p99 settlement | 178 ms | 2,577 ms | 14.5× 更快 |
| 成功率 | 94.7% | 99.9% | Base 更稳 |
| 失败时的耗时（中位） | 154 ms（快速 fail） | 17.8 秒（gas estimation retry storm） | Solana 失败不堵塞 |

**体感差距最大的场景：流式聊天的"第一个 chunk"延迟** ——
- **Solana**：结算 ~106ms + 上游开始返 ≈ 首 chunk **~150ms**
- **Base**：结算 ~935ms + 上游开始返 ≈ 首 chunk **~1.1s**

**结论：**
- ✅ **Solana**：燃料费 < $0.001、首 chunk 快 9 倍、和 Base 完整功能对齐（含 tools 调用、async 流式，自 `blockrun-litellm 0.3.2`）
- ✅ **Base**：稳一些（99.9% vs 94.7%），USDC 流动性大，适合作为 Solana 故障时的 fallback

**推荐策略**：默认走 Solana，配 LiteLLM router fallback 切到 Base 应对偶发失败：

```yaml
router_settings:
  num_retries: 2
  fallbacks:
    - {"blockrun-solana-*": ["blockrun-base-*"]}   # Solana 挂了切 Base
```

切换两边只是改 `api_base` 和 `api_key`，业务代码不用动。（Solana 的 image / Exa / Predexon 这些端点目前只支持同步，chat 已经完整异步 + 流式。）

---

## 链接

- 📦 PyPI：https://pypi.org/project/blockrun-litellm/
- 🌐 GitHub：https://github.com/BlockRunAI/blockrun-litellm
- 🛠 底层 SDK：https://github.com/BlockRunAI/blockrun-llm
- 🌐 BlockRun 官网：https://blockrun.ai

问题请提 issue：https://github.com/BlockRunAI/blockrun-litellm/issues（一般 24 小时内回复）
