"""Configuration management."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# Database configuration
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_DATABASE = os.environ.get("PG_DATABASE", "nooshare_dev")
PG_SSLMODE = os.environ.get("PG_SSLMODE", "prefer")

# Indicator parameters
VOLATILITY_PERIODS = [5, 10, 20, 60]
MOMENTUM_PERIODS = [5, 10, 20, 60]
VOLUME_PERIODS = [5, 10, 20]
PRICE_POSITION_PERIODS = [5, 10, 20, 60]

# Technical indicator defaults
BOLL_PERIOD = 20
BOLL_STD = 2.0
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14

# Data range
MIN_DATE = 20190101

# 统一行情接口：固定复权因子版本（P4）。
# indicators 只读指定 factor_set_id 的 market_daily_adjusted_bar，
# 保证同一代码版本在不同时间读到同一 HFQ 序列（可复现/PIT）。
# v1 = 11111111-1111-4111-8111-111111111111；DEV 2026-11-28 A/B 500股×全历史
# 877,990 行 raw/hfq/volume/amount 全 0 mismatch，全量行数一致。
MARKET_FACTOR_SET_ID = os.environ.get(
    "MARKET_FACTOR_SET_ID", "11111111-1111-4111-8111-111111111111"
)
