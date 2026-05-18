# BlockRun 企业问答 — 限流 / 部署 / SLA

> 验证日期：2026-05-18
> 适用于：评估 BlockRun 用于生产环境的 enterprise 采购方
>
> 本页回答的三个高频问题：
> 1. **限流** — QPS / RPM / TPM / TPH 怎么算？全模型统一还是按 model 分？
> 2. **部署 / 扩容** — 升级或扩容需要多久？会中断服务吗？
> 3. **SLA** — 故障响应、可用性、值守覆盖是什么标准？

联系：`vicky@blockrun.ai`、Telegram `@bc1max`

---

## 1. 限流标准（QPS / RPM / TPM / TPH）

**BlockRun 平台层不设 QPS / RPM / TPM / TPH 配额。** 我们是 pay-per-call 模型，付费即用，没有按月调用配额。有效限流来自**上游 provider 的—**，按 **provider 区分**（不按 model 区分）。

### 各 provider 的参考限流

下面是当前各上游 provider 给我们的实际上限。**非合同条款**，会随我们与 provider 重新分级而变化，按"数量级参考"而不是"SLA"对待：

| Provider | 典型 RPM / model | 典型 TPM / model | 备注 |
|---|---|---|---|
| OpenAI (tier 5) | ~10,000 | ~2M | — |
| Anthropic | ~4,000 | ~400K | — |
| Google Gemini | ~1,000 | ~4M | — |
| DeepSeek | ~10,000+ | 宽松 | — |
| xAI (Grok) | ~3,000 | ~1M | — |
| Moonshot / MiniMax / ZAI | 各家差异 | 各家差异 | 当前流量远低于上限 |
| NVIDIA（免费 model） | ~60 RPM per IP | — | NVIDIA 侧按源 IP 限流，高并发建议改用付费 model |
| Google（图像：nano-banana / nano-banana-pro） | ~60 RPM | 不适用 | 生图耗时 3–8 秒，实际并发受单次生成时长限制 |
| OpenAI（图像：dall-e-3 / gpt-image-1/2） | ~50 RPM | 不适用 | — |
| xAI（图像：grok-imagine） | ~60 RPM | 不适用 | — |
| ZAI（图像：cogview-4） | 按 provider | 不适用 | — |
| token360（视频） | 按 model 不同 | 按 model 不同 | Seedance 异步生成，限流通常表现为排队等待而非 429 |
| Suno（音乐） | 按 provider | 不适用 | 按任务配额 |
| Bland.ai（语音） | 按 provider | 不适用 | 按账户并发上限 |

### 撞上游限流时的响应

收到 **HTTP 429** 时网关附带：

- **`Retry-After`** 头（RFC-7231 标准）—— 重试前需等待的秒数；从上游错误中解析，缺失时默认 60 秒
- **`X-RateLimit-Source: <provider>`** 头 —— 告诉你是哪个上游 provider 限流了
- Body 含 `code: "RATE_LIMITED"` + `source` + `retry_after_seconds`

客户端策略二选一：
- **(a) 同 provider 重试** —— 等 `Retry-After` 秒后重试同 model
- **(b) Fail-over 到同级别其他 provider 的 model** —— 如 OpenAI 限流时切到 `anthropic/claude-sonnet-4.6`

### 网关层 IP 限流（仅 metadata 端点）

只有 metadata / 发现端点有 IP 级限流：

| 端点 | 限制 |
|---|---|
| `GET /v1/models`、`/v1/{image,video,audio}/models` | 100 req / 小时 per IP |
| `GET /api/pricing` | 100 req / 小时 per IP |
| `GET /api/health/*` | 60 req / 分钟 per IP |
| `GET /api/v1/voice/call/{id}`（poll） | 无网关限流 |

**结论：限流按上游 provider 区分，不按 BlockRun 端的 model 单独区分。**

### 需要保证 QPS / TPM？

走 enterprise 路径：
- **专属 key pool**（独占 OpenAI / Anthropic / Google 等 key，不和共享池竞争）
- **预留 provider TPM / RPM**（向 provider 加预付保留容量，写入合约）
- **自定义 429 行为**（白名单 IP 永不被限）

完整规则：https://blockrun.ai/docs/api-reference/rate-limits

---

## 2. 部署 / 扩容时长

部署架构：GCP Cloud Run（serverless container），前端 Cloud Load Balancer + CDN。所有容量调整都是零中断。

**两套网关，分别部署：**

| 网关 | 链 | Region |
|---|---|---|
| `blockrun.ai` | Base（USDC on Base） | us-central1（单 region） |
| `sol.blockrun.ai` | Solana（USDC on Solana） | us-central1 + **asia-northeast1**（双 region，Anycast LB 就近路由） |

### 操作与时长表

| 操作 | 完成时长 | 客户感知影响 |
|---|---|---|
| **代码升级（新 revision）** | **5–10 分钟（端到端）** | **零中断** —— 健康检查通过后 atomic traffic flip；失败自动 rollback |
| **水平扩容（增加实例）** | **<10 秒** | 零中断 —— Cloud Run 按并发请求量自动 scale 0 → N |
| **冷启动延迟**（落到新实例的第一个请求） | ~1–3 秒 | 仅影响新实例第一个请求，后续都是热路径 |
| **消除冷启动**（设 `--min-instances=N`） | <1 分钟生效 | 零中断 |
| **配置 / 密钥更新**（env / secrets） | ~2 分钟 | 零中断 —— 新 revision atomic switch |
| **多 region 扩展**（如加 Asia / EU 边缘） | 1–2 小时 | 零中断 —— Anycast LB 接入新 region 不中断流量 |
| **数据库 / Firestore failover**（罕见） | <5 分钟（GCP 托管） | 钱包归属查询短暂只读降级；支付流程不受影响 |

### 发布流水线

每次生产部署走 `./deploy-safe.sh`：

1. **Build** —— Cloud Build 构建 Docker 镜像（~3–4 分钟）
2. **部署到 0% 流量的新 revision** —— 新代码 live 但还没有流量（~30 秒）
3. **跑健康检查：**
   - `GET /api/health` —— 基本存活
   - `GET /api/v1/models` —— 模型注册表加载正确
   - **真实付费 API 调用** —— 验证端到端支付 + provider 路由
4. **流量 atomic 切换到 100%** —— Cloud Run 流量从 `0% 新 / 100% 旧` 瞬间切到 `100% 新 / 0% 旧`
5. **任何健康检查失败** —— 中止切换，继续从旧 revision 服务（流量根本没切过去，所以也不需要 rollback）

### 自动扩缩容行为

Cloud Run 按 in-flight 并发请求数扩容：

- **每实例并发目标：** 80 个并发请求
- **最小实例数：** 1（生产网关保留 1 个常驻，避免冷启动）
- **最大实例数：** 100（硬上限防止失控成本；可按需提升）
- **扩容触发：** 队列深度超过并发目标 → 新实例启动（~1–3 秒）
- **缩容触发：** 实例空闲 15 分钟 → 终止

10× 流量峰值秒级被吸收。

### 容量上限

| 资源 | 默认上限 |
|---|---|
| 网关并发请求 | 100 实例 × 80 并发 = **8,000 个同时 in-flight** |
| 上游 provider 吞吐 | 受限于各家 RPM/TPM（见第 1 节）|
| 构建队列并行度 | 1 个环境同时只 ship 一次 |

### 地理延迟

**blockrun.ai（Base，单 region us-central1）：**

| 起源地 | 中位延迟 |
|---|---|
| 美国 | 20–60 ms |
| 西欧 | 110–140 ms |
| 亚洲 HK / SG / JP | 180–230 ms |
| 亚洲 KR / CN | 200–260 ms |

**sol.blockrun.ai（Solana，双 region us-central1 + asia-northeast1）：**

| 起源地 | 中位延迟 |
|---|---|
| 美国 | 20–60 ms |
| 西欧 | 110–140 ms |
| 亚洲 HK / SG / JP / KR | **30–60 ms**（asia-northeast1 就近） |
| 亚洲 CN | 60–120 ms |

欧洲延迟敏感部署 → enterprise 合约可加 `europe-west4` 副本，Anycast LB 自动就近。

---

## 3. SLA / 服务等级

| 指标 | 当前承诺 | Enterprise 合约可承诺 |
|---|---|---|
| **平台可用性** | ~99.9%（过去 90 天 Cloud Run SLO 实测） | **99.9% 合约级 SLA**，含按比例 credit 退款条款 |
| **故障首次响应** | 工作时段（亚洲 + 美国 09:00–22:00 PT）<2 小时 | **4 小时合约级首响**（任何时段） |
| **故障处理 / 恢复** | P0 全平台不可用 <2 小时；P1 单 provider / 单 endpoint <24 小时；P2 非阻塞 <5 工作日 | **签订时按等级条款固化** |
| **支持渠道** | Telegram `@bc1max`（主要） + email `vicky@blockrun.ai` | 专属 Slack 频道 + on-call 工程师 pager + 季度 review |
| **7×24 值守** | 覆盖亚洲 + 美国工作时段 | 7×24 on-call rotation（仅 enterprise 合约） |

### 故障等级定义

| 等级 | 定义 | 首响目标 | 解决目标 |
|---|---|---|---|
| **P0** | 全平台 / 主要 endpoint 不可用，影响所有客户 | 15 分钟 | 2 小时 |
| **P1** | 单 provider / 单 endpoint 异常，部分客户受影响 | 1 小时 | 24 小时 |
| **P2** | 非阻塞 bug 或退化，有 workaround | 1 工作日 | 5 工作日 |
| **P3** | 功能改进 / nice-to-have | 1 工作日 | 排期 |

### 状态与可观测性

故障判定原则：看到某 model 持续 5xx，请先联系支持确认是上游 provider 降级还是网关问题 —— 大多数情况是前者。

Enterprise 合约下开放：
- Cloud Logging 看板（per-route P50/P95/P99 延迟、per-provider 4xx/5xx 比例）
- GCS 镜像的请求/响应日志（含 PII 脱敏）用于事后追溯

### Enterprise 升级路径

如贵团队的生产部署需要：
- 99.9% 合约级 SLA + 退款条款
- 7×24 on-call
- 专属 key pool / 保留 provider TPM
- 专属 Slack channel + 季度 review
- VPC peering / 自托管选项

按月度 commit 报价，单独安排电话聊需求。

**联系：** `vicky@blockrun.ai` · Telegram `@bc1max`

---

## 相关文档

- [CUSTOMER-ONBOARDING.zh.md](./CUSTOMER-ONBOARDING.zh.md) — 5 分钟把 BlockRun 接入 LiteLLM
- [PROXY-FULL-SETUP.zh.md](./PROXY-FULL-SETUP.zh.md) — LiteLLM Proxy Server + UI 完整部署
- [限流英文版（公开）](https://blockrun.ai/docs/api-reference/rate-limits) — Rate Limits docs
