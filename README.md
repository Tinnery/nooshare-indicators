# nooshare-indicators

A-share daily technical indicator computation engine.

## Features

- **收益率指标**: 简单收益率、对数收益率（含可加性验证）
- **波动率指标**: 4 种估计量（Close-to-Close、Parkinson、Garman-Klass、Rogers-Satchell）
- **动量/趋势指标**: RSI、MACD、ROC、KAMA、EMA、SMA、DEMA
- **成交量指标**: VWAP、OBV、Volume Ratio、Volume Z-Score
- **价格位置指标**: 布林带、ATR、N日新高/新低、均线偏离度
- **截面标准化**: 每个指标同时输出原始值 + 截面 rank + 截面 zscore

## 数据源

所有数据来自 pro 主机 PostgreSQL `nooshare_dev` 数据库：

- `tong.stock_daily` / `tong.index_daily` / `tong.etf_daily` — 不复权日线 OHLCV
- `tong.gbbq` — 除权除息事件
- `tong.symbol_meta` — 证券分类
- `ju.stocks` — 股票列表（~5,541只）
- `ju.trading_day` — 交易日历（~4,550日）
- `ju.st_period` — ST 状态区间
- `ju.limit_ratio` — 涨跌停比例

## 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# 可选：安装 TA-Lib（用于验证）
pip install TA-Lib
```

## 配置

创建 `.env` 文件：

```env
PG_HOST=pg.pro.neuralj.com
PG_PORT=5432
PG_USER=travis
PG_PASSWORD=your_password
PG_DATABASE=nooshare_dev
PG_SSLMODE=prefer
```

## 使用方法

### 1. 初始化数据库

```bash
indicators init-db
```

创建 `indicators` schema 和所有表：
- `indicators.meta` — 指标注册表
- `indicators.daily_base_aligned` — 对齐后的基础行情
- `indicators.daily_volatility` — 波动率指标
- `indicators.daily_momentum` — 动量指标
- `indicators.daily_volume_indicators` — 成交量指标
- `indicators.daily_price_position` — 价格位置指标
- `indicators.cross_section_stats` — 截面统计

### 2. 计算所有指标

```bash
# 计算所有股票的所有指标
indicators compute

# 指定起始日期
indicators compute --start-date 20200101

# 限制股票数量（测试用）
indicators compute --max-stocks 100

# 不计算截面指标
indicators compute --no-cross-section

# 不保存到数据库（仅计算）
indicators compute --no-save
```

### 3. 测试单只股票

```bash
indicators test-stock 000001
indicators test-stock 600000 --start-date 20230101
```

### 4. 验证（需要 TA-Lib）

```bash
indicators validate
```

## 指标详细说明

### 收益率指标

| 指标 | 公式 | 预热期 |
|------|------|--------|
| simple_return | (close - pre_close) / pre_close | 1 |
| log_return | ln(close / pre_close) | 1 |

**对数收益可加性**: `sum(log_return[t:t+n]) == log(close[t+n]/close[t])`

### 波动率指标

| 指标 | 公式核心 | 效率 | 适用场景 |
|------|---------|------|---------|
| vol_ccc | std(log_returns) | 1.0x | 基准 |
| vol_parkinson | ln(high/low) 极差 | 5.1x | 无漂移 |
| vol_gk | OHLC 加权方差 | 7.4x | 最常用 |
| vol_rs | 带漂移极差 | 5.1x | 有趋势 |

参数: [5, 10, 20, 60] 日

### 动量/趋势指标

| 指标 | 公式 | 参数 |
|------|------|------|
| SMA | mean(close, period) | [5,10,20,60] |
| EMA | alpha=2/(period+1) | [5,10,20,60] |
| DEMA | 2*EMA - EMA(EMA) | [5,10,20,60] |
| KAMA | 自适应 ER | [10,20,60] |
| RSI | 100 - 100/(1+RS) | [14] |
| MACD | EMA(12)-EMA(26), Signal=9 | (12,26,9) |
| ROC | (close-close.shift(n))/close.shift(n) | [10,20] |

### 成交量指标

| 指标 | 公式 | 参数 |
|------|------|------|
| VWAP | sum(TP*vol)/sum(vol) | [5,10,20] |
| OBV | sign(delta)*volume 累积 | — |
| volume_ratio | volume/avg_volume | [5,10,20] |
| volume_zscore | (vol-mean)/std | [5,10,20] |

### 价格位置指标

| 指标 | 公式 | 参数 |
|------|------|------|
| bollinger | SMA ± 2*std | (20,2) |
| ATR | Wilder 平滑 TR | (14) |
| n_day_high | rolling.max(high) | [5,10,20,60] |
| n_day_low | rolling.min(low) | [5,10,20,60] |
| ma_deviation | (close-SMA)/SMA*100 | [5,10,20,60] |

## 截面标准化

每个指标同时输出：
- `value`: 原始时序值
- `cross_section_rank`: 当日全市场百分位排名（0-1）
- `cross_section_zscore`: 当日全市场 Z-Score

**为什么需要截面标准化**：
- "RSI=70" 本身没有意义
- "RSI 在全市场排 95%" 告诉你：这只股票比 95% 的股票更超买

## 计算约束

1. **严格无未来函数**: T 日指标只使用 ≤T 数据
2. **涨跌停标记**: `is_censored` 标记流动性截断
3. **ST 处理**: 使用 `ju.limit_ratio` 的实际比例
4. **预热期**: 数据不足时输出 NULL
5. **交易日历对齐**: 停牌日 fill forward + volume=0

## 验证标准

| 验证项 | 方法 | 通过条件 |
|--------|------|---------|
| TA-Lib 一致性 | 与 TA-Lib C 库逐值比对 | max abs error < 1e-6 |
| 收益可加性 | sum(log_return[t:t+n]) == log(close[t+n]/close[t]) | 误差 < 1e-10 |
| 复权一致性 | hfq_close[t] = raw_close[t] * cumulative_factor | 与 tong 数据交叉验证 |
| 涨跌停一致性 | 与 neural-base 的 limit 逻辑比对 | 100% 一致 |
| 截面完整性 | 每交易日非停牌股票都有值 | 缺失率 < 0.1% |

## 项目结构

CLI-driven application pattern: **CLI → Task → Units → Adapters**

```
nooshare-indicators/
├── nooshare_indicators/
│   ├── __init__.py            # Package marker
│   ├── cli.py                 # CLI trigger (entry point)
│   ├── config.py              # Configuration & constants
│   ├── tasks/                 # Application orchestration
│   │   ├── compute.py         # Full-pipeline & single-stock computation
│   │   └── validate.py        # TA-Lib cross-validation
│   ├── indicators/            # Business units (pure computation)
│   │   ├── returns.py         # Log/simple returns
│   │   ├── volatility.py      # CCC, Parkinson, Garman-Klass, Rogers-Satchell
│   │   ├── momentum.py        # SMA, EMA, DEMA, KAMA, RSI, MACD, ROC
│   │   ├── volume.py          # VWAP, OBV, Volume Ratio, Volume Z-Score
│   │   ├── price_position.py  # Bollinger, ATR, N-day High/Low, MA Deviation
│   │   └── cross_section.py   # Cross-sectional rank & zscore
│   └── infrastructure/        # Adapters (database, data loading)
│       ├── database.py        # PostgreSQL connection & operations
│       └── data_loader.py     # Trading calendar, stock data, batch loading
├── tests/                     # Test package
├── deliverables/              # 验证报告等交付物
├── .env                       # 环境变量
├── pyproject.toml
└── README.md
```

## 技术栈

- Python 3.10+
- pandas + numpy — 向量化计算
- psycopg2 — PostgreSQL 连接
- TA-Lib（可选）— 验证基准

## 性能优化

- 使用 COPY + 临时表 upsert（沿用 neural-base 模式）
- 批量写入，避免逐行 INSERT
- 向量化计算，避免逐行循环
- 支持增量计算（通过 start_date 参数）

## 路线图

- [ ] 实现 TA-Lib 验证逻辑
- [ ] 添加更多指标（如 VWAP、TWAP、实现波动率等）
- [ ] 支持分钟级指标
- [ ] 添加回测框架集成
- [ ] 实现指标因子分析

## 许可证

MIT
