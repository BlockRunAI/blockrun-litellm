# LiteLLM Proxy + BlockRun —— 完整部署（含 UI）

> 实测日期：2026-05-15，macOS，Python 3.13，Docker Desktop 28.4，
> LiteLLM 1.83.x，blockrun-litellm 0.3.5，blockrun-llm 0.24.1。
> 下面每条命令都是真人跑过、输出真实贴出。
>
> 这份是**完整部署**：LiteLLM Proxy Server + 管理 UI、Postgres 存
> 元数据/spend/keys、BlockRun sidecar 做 x402 签名、JSONL 请求日志。
> 适合想要 LiteLLM 管理面板（`http://localhost:4000/ui`）的场景。
>
> 只要 API 访问（不要 UI、不要 DB）？看
> [`CUSTOMER-ONBOARDING.zh.md`](./CUSTOMER-ONBOARDING.zh.md) —— 更简单，不需要 Docker。

---

## 完成后能拿到什么

10 分钟后你会有：

```
┌────────────┐    HTTPS API     ┌────────────────────┐    OpenAI 兼容 HTTP        ┌──────────────────┐    HTTPS + x402    ┌─────────────────────┐
│ 你的应用   │─────────────────▶│ LiteLLM Proxy       │───────────────────────────▶│ blockrun-litellm │───────────────────▶│ Solana 网关         │
│ 或 curl   │  Bearer key      │ + 管理 UI           │  api_base=http://4001       │ -proxy (sidecar)│   PAYMENT-SIGNATURE│ sol.blockrun.ai     │
│            │                  │ 端口 4000           │                            │ 端口 4001        │                    │ 或 blockrun.ai      │
└────────────┘                  └────────────────────┘                            └──────────────────┘                    └─────────────────────┘
                                          │                                              │
                                          │ Prisma                                       │ 本地 Ed25519 / EIP-712 签名
                                          ▼                                              ▼
                                  ┌──────────────────┐                          ┌──────────────────────┐
                                  │ Postgres 16      │                          │ $SOLANA_WALLET_KEY   │
                                  │ 端口 5544 docker │                          │ $BLOCKRUN_WALLET_KEY │
                                  │ 卷 litellm-pg    │                          │ ~/.blockrun/.session │
                                  └──────────────────┘                          └──────────────────────┘
```

**UI 能看到：** 模型列表、请求日志、单笔用量、按 key/模型/天聚合 spend、虚拟子 key（带预算 / 模型白名单 / 限速）、API key 管理、Settings。

---

## 前置准备

| 需要 | 用途 |
|---|---|
| Python ≥ 3.9（用 3.13 测的） | 跑 LiteLLM + sidecar |
| Docker Desktop | Postgres for LiteLLM UI |
| **Solana 钱包**（推荐）或 Base 钱包，有几个 USDC | 按次付费 |
| 端口 4000、4001 空闲 | Proxy 和 sidecar |
| 端口 5544（或其他空闲端口）给 Postgres | 5432 通常被本地 Postgres.app 占了 |

---

## Step 1 —— 安装 Python 包

```bash
python3.13 -m venv ~/.venv/litellm-blockrun
source ~/.venv/litellm-blockrun/bin/activate

pip install -U 'litellm[proxy]' 'blockrun-litellm[proxy,solana]' prisma
```

`prisma` 包**必装** —— LiteLLM Proxy 的 UI/DB 功能要它，但是 `litellm[proxy]` 不会自动拉。

`blockrun-litellm[proxy,solana]` 装：BlockRun adapter、sidecar、Solana x402 SVM 工具链。

---

## Step 2 —— 配置钱包（一次性）

**陛下推荐：用 Solana 钱包。** 燃料费 < $0.001/笔，几秒终态，功能与 Base 完全对齐。

### Solana（首选）

```bash
# 方式 A：已有 Solana base58 私钥
export SOLANA_WALLET_KEY=你的BASE58_SOLANA_私钥

# 方式 B：自动生成 + 二维码引导充值 USDC
python -c "from blockrun_llm import setup_agent_solana_wallet; setup_agent_solana_wallet()"
# → 保存到 ~/.blockrun/.solana-session，下次自动加载
```

### Base（备选）

```bash
export BLOCKRUN_WALLET_KEY=0x你的Base私钥
# 或者自动生成：
python -c "from blockrun_llm import setup_agent_wallet; setup_agent_wallet()"
```

> 私钥**永不离开你的机器** —— 只用于本地签名，签名才上链。

> 💡 想免费测一遍？用模型 `nvidia/deepseek-v4-flash` —— 同样的流程，零结算。

---

## Step 3 —— 启动 BlockRun sidecar

**终端 1**：

```bash
source ~/.venv/litellm-blockrun/bin/activate
export SOLANA_WALLET_KEY=...                        # 用 Solana

blockrun-litellm-proxy --port 4001 --api-url https://sol.blockrun.ai/api
# → INFO:     Uvicorn running on http://127.0.0.1:4001
```

如果用 Base：

```bash
export BLOCKRUN_WALLET_KEY=0x...
blockrun-litellm-proxy --port 4001
```

验证：

```bash
curl http://127.0.0.1:4001/healthz
# → {"status":"ok"}
```

---

## Step 4 —— 起 Postgres（UI 必需）

LiteLLM 管理 UI **必须**要 Postgres —— 没数据库登录都不会处理，返
`Authentication Error, Not connected to DB!`。本地 Docker 起一个。

**⚠️ 5432 端口冲突坑**：很多 Mac 已经有 Postgres.app 或 Homebrew
postgres 在 5432，Docker 端口映射"看起来成功"，但主机 `psql` 实际连
到的是主机的 Postgres 不是容器。用 **5544** 这种空闲端口。

```bash
# 检查 5432 是否被占（一般是 Postgres.app）：
lsof -nP -iTCP:5432 | grep LISTEN

# 起 Postgres 在 5544：
docker run -d --name litellm-pg \
  -e POSTGRES_PASSWORD=litellm \
  -e POSTGRES_USER=litellm \
  -e POSTGRES_DB=litellm \
  -p 5544:5432 \
  postgres:16

# 等连接就绪（一般 < 2 秒）：
until docker exec litellm-pg pg_isready -U litellm 2>/dev/null | grep -q "accepting"; do
  sleep 1
done
echo "Postgres ready on 127.0.0.1:5544"
```

### 把 LiteLLM 的 schema 推到 DB

LiteLLM 自带 Prisma schema，跑一次同步：

```bash
# 定位 LiteLLM 的 schema 目录
SCHEMA_DIR=$(python -c "import litellm.proxy, os; print(os.path.dirname(litellm.proxy.__file__))")
cd "$SCHEMA_DIR"

# 生成 Prisma client（每个 venv 一次）
prisma generate

# 推 schema 到 Postgres
DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:5544/litellm \
  prisma db push --accept-data-loss --skip-generate
# → "🚀  Your database is now in sync with your Prisma schema."

cd -
```

如果遇到 `Error: P1010: User 'litellm' was denied access on the
database 'litellm.public'`，Postgres 15+ 默认 schema 权限不够，
给用户加 superuser：

```bash
docker exec litellm-pg psql -U litellm -d litellm \
  -c "ALTER USER litellm WITH SUPERUSER;"
# 然后重跑 prisma db push
```

---

## Step 5 —— 写 LiteLLM 配置

选个工作目录（如 `~/litellm-blockrun-deploy/`）—— **`custom_callbacks.py` 桥接文件必须放在这个目录**，因为 LiteLLM Proxy 用相对文件名加载 callback，不读已安装的包。

```bash
mkdir -p ~/litellm-blockrun-deploy
cd ~/litellm-blockrun-deploy
```

### `config.yaml`

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: openai/openai/gpt-5.5            # 双 'openai/' 是故意：
                                              #   前者 = LiteLLM provider（"OpenAI 兼容 HTTP"）
                                              #   后者 = BlockRun 模型 id
      api_base: http://127.0.0.1:4001/v1      # ← BlockRun sidecar
      api_key: "dummy"                        # 不用；sidecar 负责认证

  - model_name: claude-opus-4-7
    litellm_params:
      model: openai/anthropic/claude-opus-4-7
      api_base: http://127.0.0.1:4001/v1
      api_key: "dummy"

  - model_name: deepseek-v4-flash             # 免费 —— 烟雾测试首选
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
  drop_params: True                            # 静默丢弃 BlockRun 不支持的 OpenAI 参数
  callbacks: ["custom_callbacks.blockrun_logger"]   # JSONL 请求日志

general_settings:
  master_key: "sk-blockrun-demo-master"        # ← 生产环境务必改！
                                               # 同时也是 UI 默认密码（admin / <master_key>）
```

### `custom_callbacks.py`

```python
# 放在 config.yaml 同目录。
# LiteLLM Proxy 用相对文件名加载 callback，不读已安装的 PyPI 包，
# 所以这两行桥接文件是必要的（即使 blockrun_litellm.logger 已在 import path）。
from blockrun_litellm.logger import JSONLLogger

blockrun_logger = JSONLLogger()
```

---

## Step 6 —— 启动 LiteLLM Proxy

**终端 2**（sidecar 必须在终端 1 还跑着）：

```bash
source ~/.venv/litellm-blockrun/bin/activate
cd ~/litellm-blockrun-deploy

export DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:5544/litellm
export BLOCKRUN_LITELLM_LOG=~/litellm-blockrun-deploy/calls.jsonl

litellm --config config.yaml --port 4000 --host 127.0.0.1
```

应该看到：

```
   ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
   ...
LiteLLM: Proxy initialized with Config, Set models:
    gpt-5.5
    claude-opus-4-7
    deepseek-v4-flash
    gemini-3.1-pro
INFO:     Uvicorn running on http://127.0.0.1:4000
```

验证：

```bash
curl http://127.0.0.1:4000/health/liveliness
# → {"status":"healthy"}

curl -L http://127.0.0.1:4000/ui
# → 200（HTML 页面）
```

---

## Step 7 —— 打开 UI

浏览器 → **http://127.0.0.1:4000/ui/**

| 字段 | 值 |
|---|---|
| Username | `admin` |
| Password | `sk-blockrun-demo-master`（即 `master_key`） |

> 登录页提示 "Password is your set LiteLLM Proxy MASTER_KEY." ——
> 字面意思 —— `config.yaml` 的 `master_key` 就是 UI 默认密码。要单独
> 设 UI 密码，用环境变量 `UI_PASSWORD=...`（`config.yaml` 里的
> `ui_password` 字段**会被忽略** —— LiteLLM 已知小坑）。

UI 里能做：

- **Models** —— 看注册的 4 个模型，浏览器里直接测调用
- **Virtual Keys** —— 生成子 key（含单 key 预算 / 模型白名单 / 限速）。给每个客户/团队各发一个
- **Logs** —— 每条经过 proxy 的请求（DB 存）
- **Spend Analytics** —— 按 key / 模型 / 天聚合 spend
- **Settings** —— 缓存、告警等

---

## Step 8 —— 烟雾测试

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-blockrun-demo-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"说一个词"}],
    "stream": true
  }'
```

预期：`text/event-stream` 响应，几个 `data: {...}` chunks 加 `data: [DONE]`。

**图像生成**（直接打 sidecar，v0.3.6 起）：

```bash
curl http://127.0.0.1:4001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "google/nano-banana", "prompt": "一只穿着宇航服的柯基", "size": "1024x1024"}'
```

或通过 LiteLLM：

```python
import litellm
resp = litellm.image_generation(
    model="openai/google/nano-banana",
    prompt="一只穿着宇航服的柯基",
    api_base="http://127.0.0.1:4001/v1",
    api_key="dummy",
)
print(resp.data[0].url)
```

然后查日志：

```bash
tail -1 ~/litellm-blockrun-deploy/calls.jsonl | python -m json.tool
```

会看到一行 JSON：`model`、`messages`、`completion`、`usage`、`latency_ms`、`status` 等 ——
完整 schema 见 [`CUSTOMER-ONBOARDING.zh.md`](./CUSTOMER-ONBOARDING.zh.md)。

---

## 所有持久化文件一览

| 文件 / 环境变量 | 内容 | 可改吗 |
|---|---|---|
| `$SOLANA_WALLET_KEY` | Solana 私钥（推荐） | 是 |
| `$BLOCKRUN_WALLET_KEY` | Base 私钥 | 是 |
| `~/.blockrun/.solana-session` | 自动生成的 Solana 钱包 | — |
| `~/.blockrun/.session` | 自动生成的 Base 钱包 | — |
| `~/.blockrun/cost_log.jsonl` | 按笔 USDC 成本（SDK） | — |
| `~/.blockrun/data/*.json` | 付费调用完整请求/响应归档 | — |
| `~/litellm-blockrun-deploy/calls.jsonl` | LiteLLM 请求日志（适配层） | `BLOCKRUN_LITELLM_LOG` env |
| Docker 卷 `litellm-pg` | LiteLLM keys / spend / logs（Postgres） | `DATABASE_URL` env |
| `$DATABASE_URL` | LiteLLM Proxy DB 连接串 | 是 |
| `$BLOCKRUN_PROXY_TOKEN` | sidecar 可选 Bearer 鉴权 | 是 |

---

## 启停 / 重启

```bash
# 停（按顺序）
kill %1   # litellm 那个终端
kill %1   # blockrun-litellm-proxy 那个终端
docker stop litellm-pg

# 重新启
docker start litellm-pg                              # Postgres 数据保留
blockrun-litellm-proxy --port 4001 --api-url https://sol.blockrun.ai/api &
DATABASE_URL=... litellm --config config.yaml --port 4000
```

要彻底清空：

```bash
docker rm -f litellm-pg          # 销毁数据库
rm -rf ~/.blockrun/                # ⚠️ 删钱包 —— 先备份私钥！
```

---

## 踩坑与解决（都是实测踩出来的）

### `Authentication Error, Not connected to DB!`（UI 登录时）

`DATABASE_URL` 没设，或 Postgres 连不上。验证：

```bash
PGPASSWORD=litellm psql -h 127.0.0.1 -p 5544 -U litellm -d litellm -c "SELECT 1"
```

挂了说明 Postgres 容器没起，或撞上下面的端口冲突坑。

### 5432 端口冲突 —— 主机已有 Postgres

`docker run -p 5432:5432 postgres` 看起来成功，但主机 `psql -h 127.0.0.1` 实际连的是**主机的** Postgres 不是容器。症状：`role "litellm" does not exist`，可是你明明在容器里建过这个用户。

解决：用空闲端口如 5544（上文已经这么做了）。

### `Unable to find Prisma binaries. Please run 'prisma generate' first.`

Step 4 漏跑 `prisma generate`。从 LiteLLM 的 schema 目录跑：

```bash
SCHEMA_DIR=$(python -c "import litellm.proxy, os; print(os.path.dirname(litellm.proxy.__file__))")
cd "$SCHEMA_DIR" && prisma generate
```

### `ModuleNotFoundError: No module named 'prisma'`

`litellm[proxy]` 不带 `prisma` 包。手动装：`pip install prisma`。

### `P1010: User 'litellm' was denied access on the database 'litellm.public'`

Postgres 15+ schema 权限变严了。容器里给 superuser：

```bash
docker exec litellm-pg psql -U litellm -d litellm \
  -c "ALTER USER litellm WITH SUPERUSER;"
```

### `Invalid credentials used to access UI`

`config.yaml` 里写了 `ui_password` 没生效 —— LiteLLM 忽略这个字段。两条路：

- 用 **`master_key`** 当密码登录（默认行为），或
- 启动时 `UI_USERNAME=admin UI_PASSWORD=yours litellm --config ...` 用环境变量

### `Could not import proxy_logger from blockrun_litellm.logger`

LiteLLM Proxy **按文件名**加载 callback，不能直接引 PyPI 包。
`custom_callbacks.py` 桥接文件必须和 `config.yaml` 同目录，引用方式
`"custom_callbacks.blockrun_logger"` —— 不是
`"blockrun_litellm.logger.proxy_logger"`。

### 免费模型上游慢或 503

免费 NVIDIA NIM（`nvidia/deepseek-v4-flash` 等）上游不稳，503 和
60-90 秒响应是常事。SDK 透明退避重试 3 次（1s/2s/4s），但有时上游
真挂了。生产用付费模型，或加 router fallback：

```yaml
router_settings:
  num_retries: 2
  fallbacks:
    - {"deepseek-v4-flash": ["gpt-5.5"]}
```

### `BLOCKRUN_LITELLM_LOG` 设了但 JSONL 没写

检查路径可写、proxy 启动时真看到这个环境变量：

```bash
env | grep BLOCKRUN_LITELLM_LOG
```

用 `~` 的话展开成绝对路径或用 `$HOME` —— 某些 shell 里 `~` 在环境变量里不展开。

---

## 生产部署 checklist

上真实客户前过一遍：

- [ ] `master_key` 改成强随机值（`openssl rand -hex 32`）
- [ ] sidecar 设 `BLOCKRUN_PROXY_TOKEN`，客户端必须带 `Authorization: Bearer $TOKEN`
- [ ] sidecar 只绑 `127.0.0.1`（默认）—— 只让 LiteLLM Proxy 能访问
- [ ] 设 `UI_USERNAME` / `UI_PASSWORD` 环境变量（不要靠 master_key 当 UI 密码）
- [ ] Postgres 上托管（Neon / Supabase / RDS）—— 不要靠裸 Docker 容器
- [ ] 加 router fallback 列表，免费模型抖动不影响生产
- [ ] UI 里给每个虚拟 key 设预算上限，避免某个客户跑飞耗尽你钱包
- [ ] 钱包卫生：定期轮换、大额 USDC 冷钱包存、热钱包定期从冷钱包补
- [ ] 整套放私网内跑，或上游做 TLS 终结

---

## 服务端口对照

| 服务 | URL | 内容 |
|---|---|---|
| BlockRun sidecar 健康检查 | http://127.0.0.1:4001/healthz | `{"status":"ok"}` |
| BlockRun sidecar chat | http://127.0.0.1:4001/v1/chat/completions | 对话补全，x402 签名 |
| BlockRun sidecar 图像 | http://127.0.0.1:4001/v1/images/generations | 图像生成（0.3.6 起） |
| BlockRun sidecar 模型列表 | http://127.0.0.1:4001/v1/models | chat 模型目录 |
| LiteLLM Proxy 健康 | http://127.0.0.1:4000/health/liveliness | `{"status":"healthy"}` |
| LiteLLM Proxy API | http://127.0.0.1:4000/v1 | 你的应用调这里 |
| LiteLLM Proxy UI | http://127.0.0.1:4000/ui/ | 管理面板 |
| LiteLLM Proxy `/docs` | http://127.0.0.1:4000/docs | Swagger UI |
| Postgres | postgresql://litellm:litellm@127.0.0.1:5544/litellm | DB 连接 |

---

## 链接

- LiteLLM 文档：https://docs.litellm.ai
- BlockRun：https://blockrun.ai
- blockrun-litellm GitHub：https://github.com/BlockRunAI/blockrun-litellm
- blockrun-llm SDK：https://github.com/BlockRunAI/blockrun-llm
- x402 协议：https://x402.org

问题：https://github.com/BlockRunAI/blockrun-litellm/issues
