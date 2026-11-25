# nooshare-indicators — Agent 工作规则

本仓库负责 A 股日线技术指标计算：收益率、波动率、动量、成交量、价格位置和截面标准化。

## Matrix Authority

- **nooshare-design**：系统角色与跨仓逻辑职责（architecture/matrix.md）。
- **nooshare-noos**：适用的已验证知识与工程 invariant。
- **nooshare-workenv**：工作区、主机、部署与操作授权（AGENTS.md、workspace.yaml）。

## Context Bootstrap

任务涉及以下情形时，向上索引对应权威：

| 触发条件 | 索引目标 | 读什么 |
|----------|----------|--------|
| 系统角色 / 跨仓职责变更 | nooshare-design | `architecture/matrix.md` |
| 工程 invariant / 已验证知识 | nooshare-noos | 相关 domain 文件 |
| 主机 / 部署 / 操作授权变更 | nooshare-workenv | `AGENTS.md` + `workspace.yaml` |

索引路径：通过 `workspace.yaml` 找到目标 Repo 的 checkout 路径，读取对应文件。

## Local Ownership

本仓库拥有：

- Python 源码（`nooshare_indicators/`），按职责拆分为：
  - `cli.py`：CLI 入口（trigger 层）
  - `config.py`：配置与常量
  - `tasks/`：应用编排层（compute、validate）
  - `indicators/`：业务单元（纯计算，无 I/O）
  - `infrastructure/`：适配器（数据库、数据加载）
- CLI 命令（`indicators`）
- 数据库 Schema（`indicators.*`）
- 测试（`tests/`）
- 验证报告（`deliverables/`）

上游权威不替代这些本地契约。与权威要求冲突时，向拥有该问题的权威反馈，不在本地悄悄改写权威定义。

## Current Workflow

唯一执行入口是 CLI（`nooshare_indicators/cli.py`）。

| 命令 | 状态 | 说明 |
|------|------|------|
| `init-db` | Current | 创建 `indicators` schema 和所有表 |
| `compute` | Current | 计算所有股票的所有指标（支持增量） |
| `test-stock` | Current | 测试单只股票的指标计算 |
| `validate` | Current | 与 TA-Lib 交叉验证（可选依赖） |

## 数据边界

### Read-only（上游拥有，indicators 只读）

生产代码 `infrastructure/data_loader.py` 只读 `public.*` 表，`public.*` 是标准化行情的消费入口：

| Schema | Table | 说明 |
|--------|-------|------|
| `public` | `trading_calendar` | 交易日历 |
| `public` | `canonical_symbol` | 标准化证券代码 |
| `public` | `st_status_pit` | ST 状态（时点口径） |
| `public` | `limit_rule_pit` | 涨跌停规则（时点口径） |
| `public` | `hfq_daily` | 后复权日线行情 |

价格口径：OHLC 用 HFQ（后复权），Volume/Amount 保留原始行情语义（不用复权因子调整）。依据 nooshare-design/architecture/ADR-0001-research-price-convention.md。

### Read-write（indicators 拥有）

| Schema | Table | 说明 |
|--------|-------|------|
| `indicators` | `meta` | 指标注册表 |
| `indicators` | `daily_base_aligned` | 对齐后的基础行情 |
| `indicators` | `daily_volatility` | 波动率指标 |
| `indicators` | `daily_momentum` | 动量指标 |
| `indicators` | `daily_volume_indicators` | 成交量指标 |
| `indicators` | `daily_price_position` | 价格位置指标 |
| `indicators` | `cross_section_stats` | 截面统计 |

## 计算约束

1. **严格无未来函数**：T 日指标只使用 ≤T 数据
2. **涨跌停标记**：`is_censored` 标记流动性截断
3. **ST 处理**：使用 `public.limit_rule_pit` 的实际比例
4. **预热期**：数据不足时输出 NULL
5. **交易日历对齐**：停牌日 fill forward + volume=0
6. **截面标准化**：每个指标同时输出原始值 + 截面 rank + 截面 zscore

## 测试与验证

```bash
cd nooshare-indicators
uv run pytest
```

验证标准：
- TA-Lib 一致性：max abs error < 1e-6
- 收益可加性：误差 < 1e-10
- 截面完整性：缺失率 < 0.1%

## 当前能力状态

| 能力 | 状态 | 说明 |
|------|------|------|
| 收益率指标 | ✓ 已实现 | 简单收益率、对数收益率 |
| 波动率指标 | ✓ 已实现 | 4 种估计量（CCC、Parkinson、GK、RS） |
| 动量指标 | ✓ 已实现 | RSI、MACD、ROC、KAMA、EMA、SMA、DEMA |
| 成交量指标 | ✓ 已实现 | VWAP、OBV、Volume Ratio、Volume Z-Score |
| 价格位置指标 | ✓ 已实现 | 布林带、ATR、N日新高/新低、均线偏离度 |
| 截面标准化 | ✓ 已实现 | 每指标输出 rank + zscore |
| TA-Lib 验证 | 部分实现 | 可选依赖，未全量运行 |
| 分钟级指标 | 未实现 | 路线图项 |

## 主机约束

本仓库只在 pro 主机开发和运行。禁止 clone 到 Windows 主机（travis-hp）。如果在 Windows 主机上发现本仓库的 checkout，应删除该 checkout 并修正来源（不正确的 clone 操作或 workspace 配置）。
