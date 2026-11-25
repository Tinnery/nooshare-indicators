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
