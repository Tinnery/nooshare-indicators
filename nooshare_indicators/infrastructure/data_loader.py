"""Data loading and alignment layer."""
import pandas as pd
import numpy as np
from typing import Optional
from . import database as db
from .. import config


def load_trading_calendar() -> pd.DataFrame:
    """Load trading calendar from public.trading_calendar (PIT product)."""
    sql = """
    SELECT cal_date AS trade_date
    FROM public.trading_calendar
    WHERE is_open = true
    ORDER BY cal_date
    """
    df = db.read_sql(sql)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_stock_list() -> pd.DataFrame:
    """Load stock list from public.canonical_symbol (P1 product).
    
    包含所有历史股票(包括退市股票),避免幸存者偏差。
    返回字段: stock_code, exchange, list_date, delist_date
    """
    sql = """
    SELECT code AS stock_code, exchange, list_date, delist_date
    FROM public.canonical_symbol
    ORDER BY code
    """
    return db.read_sql(sql)


def filter_stocks_by_date(
    stock_list: pd.DataFrame,
    target_date: int,
) -> list[str]:
    """Filter stock list by Point-in-Time (PIT) date.
    
    Returns stocks that were active on the target date:
    - list_date <= target_date
    - delist_date IS NULL OR delist_date >= target_date
    
    Args:
        stock_list: DataFrame with stock_code, list_date, delist_date
        target_date: Target date in YYYYMMDD format
    
    Returns:
        List of stock codes active on target_date
    """
    # Convert target_date to comparable format
    target_str = str(target_date)
    
    # Filter by list_date
    mask = stock_list["list_date"].astype(str) <= target_str
    
    # Filter by delist_date (NULL means still active)
    delist_mask = (
        stock_list["delist_date"].isna() | 
        (stock_list["delist_date"].astype(str) >= target_str)
    )
    
    filtered = stock_list[mask & delist_mask]
    return filtered["stock_code"].tolist()


def load_st_status() -> pd.DataFrame:
    """Load ST status from public.st_status_pit (P3 materialized view)."""
    sql = """
    SELECT canonical_id AS stock_code, 
           effective_from AS start_date, 
           effective_to AS end_date,
           st_type
    FROM public.st_status_pit
    ORDER BY canonical_id, effective_from
    """
    return db.read_sql(sql)


def load_limit_ratio() -> pd.DataFrame:
    """Load limit ratio from public.limit_rule_pit (PIT product).
    
    注意：limit_rule_pit 按 (security_type, exchange) 聚合，不含 stock_code。
    需要根据股票代码推断 security_type 和 exchange，然后查询对应的 limit_ratio。
    """
    # 先加载所有规则
    sql = """
    SELECT security_type, exchange, limit_ratio, effective_from, effective_to
    FROM public.limit_rule_pit
    ORDER BY security_type, exchange, effective_from
    """
    return db.read_sql(sql)


def code_to_symbol(code: str) -> str:
    """
    Convert pure stock code to symbol with exchange prefix.
    
    Examples:
        000001 -> sz000001
        600000 -> sh600000
        830799 -> bj830799
    """
    if code.startswith(("00", "30")):
        return f"sz{code}"
    elif code.startswith(("60", "68")):
        return f"sh{code}"
    elif code.startswith(("8", "4", "9")):
        return f"bj{code}"
    else:
        # Default: try to infer from database
        return f"sz{code}"


def load_stock_daily(symbol: str, start_date: Optional[int] = None) -> pd.DataFrame:
    """Load daily OHLCV for a single stock from the unified interface.
    
    使用 public.market_daily_adjusted_bar（P4 派生接口）+ 固定 factor_set_id，
    HFQ 价格(open_hfq..close_hfq)消除除权跳变；Raw 价格与 volume/amount 保留。
    口径与 market_daily_adjusted_bar 一致（DEV 2026-11-28 全量 15,655,244 行 A/B 0 mismatch）；
    冻结：volume_zscore 等计算逻辑不改。
    """
    sql = """
    SELECT symbol, date, 
           open, high, low, close,
           open_hfq, high_hfq, low_hfq, close_hfq,
           volume, amount
    FROM public.market_daily_adjusted_bar
    WHERE symbol = %s AND date >= %s AND factor_set_id = %s
    ORDER BY date
    """
    start = start_date or config.MIN_DATE
    df = db.read_sql(sql, (symbol, start, config.MARKET_FACTOR_SET_ID))

    # Convert date from integer to datetime
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


def load_all_stocks_daily_batch(
    codes: list[str],
    start_date: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load daily data for all stocks in a single query (batch mode).
    
    使用 public.market_daily_adjusted_bar + 固定 factor_set_id（口径同 load_stock_daily）。
    Returns DataFrame with all stocks' data, much faster than individual queries.
    """
    # Convert codes to symbols
    symbols = [code_to_symbol(code) for code in codes]
    
    start = start_date or config.MIN_DATE
    
    # Build query with IN clause
    placeholders = ", ".join(["%s"] * len(symbols))
    sql = f"""
    SELECT symbol, date,
           open, high, low, close,
           open_hfq, high_hfq, low_hfq, close_hfq,
           volume, amount
    FROM public.market_daily_adjusted_bar
    WHERE symbol IN ({placeholders}) AND date >= %s AND factor_set_id = %s
    ORDER BY symbol, date
    """

    df = db.read_sql(sql, tuple(symbols) + (start, config.MARKET_FACTOR_SET_ID))
    
    # Convert date from integer to datetime
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    
    # Extract code from symbol
    df["code"] = df["symbol"].str[2:]
    
    return df


def load_all_stock_daily(
    codes: list[str],
    start_date: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Load daily data for multiple stocks."""
    result = {}
    for code in codes:
        symbol = f"sz{code}" if code.startswith(("00", "30")) else f"sh{code}"
        if code.startswith(("8", "4", "9")):
            symbol = f"bj{code}"
        try:
            df = load_stock_daily(symbol, start_date)
            if not df.empty:
                result[code] = df
        except Exception as e:
            print(f"Warning: Failed to load {symbol}: {e}")
    return result


def align_with_calendar(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    code: str,
) -> pd.DataFrame:
    """
    Align stock data with trading calendar.
    Fill missing trading days with forward fill + volume=0.
    
    Handles both Raw and HFQ price columns.
    """
    if df.empty:
        return df

    # Ensure date column exists and is datetime
    if "date" not in df.columns:
        df = df.reset_index()
    
    # Convert date to datetime - handle multiple input formats
    if df["date"].dtype in ['int64', 'int32']:
        # Integer YYYYMMDD format
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    elif df["date"].dtype == 'object':
        # String format - could be ISO or YYYYMMDD
        df["date"] = pd.to_datetime(df["date"])
    # If already datetime64, no conversion needed
    
    # Set date as index
    df = df.set_index("date")

    # Convert calendar dates to datetime
    calendar_dates = pd.to_datetime(calendar["trade_date"])

    # Filter calendar to stock's date range
    min_date = df.index.min()
    max_date = df.index.max()
    
    # Get calendar dates within stock's range
    calendar_filtered = calendar_dates[
        (calendar_dates >= min_date) & (calendar_dates <= max_date)
    ]
    
    if calendar_filtered.empty:
        return df.reset_index()

    # Create a complete DataFrame with all calendar dates
    calendar_df = pd.DataFrame({"date": calendar_filtered.values})
    calendar_df = calendar_df.set_index("date")
    
    # Merge with actual data
    df = calendar_df.join(df, how="left")

    # Fill missing prices with forward fill - handle both Raw and HFQ
    price_cols = ["open", "high", "low", "close"]
    hfq_cols = ["open_hfq", "high_hfq", "low_hfq", "close_hfq"]
    
    # Fill Raw prices
    df[price_cols] = df[price_cols].ffill()
    
    # Fill HFQ prices if they exist
    for col in hfq_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Fill missing volume/amount with 0
    df[["volume", "amount"]] = df[["volume", "amount"]].fillna(0)

    # Add code column
    df["code"] = code

    # Reset index
    df = df.reset_index()

    return df


def compute_pre_close(df: pd.DataFrame) -> pd.DataFrame:
    """Compute previous day's HFQ close price for research returns.

    Per ADR-0001, research returns must use HFQ prices. The raw close stays
    in df["close"] only for limit-up/down detection against real traded
    prices; it must never silently feed research computations.

    Raises:
        ValueError: if the HFQ close column is missing.
    """
    df = df.copy()
    if "close_hfq" not in df.columns:
        raise ValueError(
            "compute_pre_close requires 'close_hfq' (HFQ basis per ADR-0001); "
            "raw close must not silently be used for research returns. "
            "Check that the upstream market_daily_adjusted_bar view provides HFQ columns."
        )
    df["pre_close"] = df["close_hfq"].shift(1)
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple and log returns on the HFQ close (research basis).

    Per ADR-0001, returns must be computed from HFQ prices so they stay
    comparable across corporate actions. Raw prices are not accepted here.

    Raises:
        ValueError: if 'close_hfq' is missing.
    """
    df = df.copy()
    if "close_hfq" not in df.columns:
        raise ValueError(
            "compute_returns requires 'close_hfq' (HFQ basis per ADR-0001); "
            "raw close must not silently be used for research returns."
        )
    close = df["close_hfq"]

    # Simple return: (close - pre_close) / pre_close
    df["simple_return"] = (close - close.shift(1)) / close.shift(1)

    # Log return: ln(close / pre_close)
    df["log_return"] = np.log(close / close.shift(1))

    return df


def mark_st_status(
    df: pd.DataFrame,
    st_periods: pd.DataFrame,
    code: str,
) -> pd.DataFrame:
    """Mark ST status for each trading day."""
    df = df.copy()
    df["is_st"] = False

    if st_periods.empty:
        return df

    # Filter ST periods for this code
    st_code = st_periods[st_periods["stock_code"] == code]

    if st_code.empty:
        return df

    # For each day, check if it falls in any ST period
    for _, row in st_code.iterrows():
        start = pd.to_datetime(row["start_date"])
        end = pd.to_datetime(row["end_date"]) if pd.notna(row["end_date"]) else pd.Timestamp.max

        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "is_st"] = True

    return df


def mark_limit_status(
    df: pd.DataFrame,
    limit_ratios: pd.DataFrame,
    code: str,
) -> pd.DataFrame:
    """Mark limit up/down status.

    Queries limit_rule_pit by (security_type, exchange) inferred from stock code.
    Uses PIT semantics: finds the rule effective on each trading date.
    """
    df = df.copy()
    df["is_limit_up"] = False
    df["is_limit_down"] = False
    df["is_censored"] = False

    if df.empty or "pre_close" not in df.columns:
        return df

    security_type, exchange = _code_to_security_type(code)

    rules = limit_ratios[
        (limit_ratios["security_type"] == security_type)
        & (limit_ratios["exchange"] == exchange)
    ].copy()

    if rules.empty:
        default_ratio = 0.05 if security_type in ("st", "bse") else 0.10
        df["limit_up_price"] = (df["pre_close"] * (1 + default_ratio)).round(2)
        df["limit_down_price"] = (df["pre_close"] * (1 - default_ratio)).round(2)
    else:
        rules = rules.sort_values("effective_from")
        rules["effective_from_dt"] = pd.to_datetime(rules["effective_from"])
        # Handle both datetime and string date formats
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["_date_dt"] = df["date"]
        else:
            df["_date_dt"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

        ratios = []
        for trade_date in df["_date_dt"]:
            effective = rules[rules["effective_from_dt"] <= trade_date]
            if not effective.empty:
                ratios.append(float(effective.iloc[-1]["limit_ratio"]))
            else:
                ratios.append(0.10)

        df["limit_ratio"] = ratios
        df["limit_up_price"] = (df["pre_close"] * (1 + df["limit_ratio"])).round(2)
        df["limit_down_price"] = (df["pre_close"] * (1 - df["limit_ratio"])).round(2)
        df = df.drop(columns=["_date_dt", "limit_ratio"])

    df["is_limit_up"] = df["close"] >= df["limit_up_price"]
    df["is_limit_down"] = df["close"] <= df["limit_down_price"]
    df["is_censored"] = df["is_limit_up"] | df["is_limit_down"]

    return df


def _code_to_security_type(code: str) -> tuple:
    """Infer (security_type, exchange) from stock code."""
    if code[:3] in ("688", "689"):
        return "star_market", "sh"
    elif code.startswith("6"):
        return "main_board", "sh"
    elif code[:3] in ("300", "301"):
        return "chi_next", "sz"
    elif code.startswith("0"):
        return "main_board", "sz"
    elif code.startswith(("8", "4")):
        return "bse", "bj"
    else:
        return "main_board", "sz"


def prepare_base_data(
    code: str,
    calendar: pd.DataFrame,
    st_periods: pd.DataFrame,
    limit_ratios: pd.DataFrame,
    start_date: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prepare aligned base data for a single stock.
    Includes: OHLCV alignment, pre_close, returns, ST/limit marks.
    """
    # Convert pure code to symbol with exchange prefix
    symbol = code_to_symbol(code)
    
    # Load raw data
    df = load_stock_daily(symbol, start_date)

    if df.empty:
        return pd.DataFrame()

    # Align with calendar
    df = align_with_calendar(df, calendar, code)

    # Compute pre_close
    df = compute_pre_close(df)

    # Compute returns
    df = compute_returns(df)

    # Mark ST status
    df = mark_st_status(df, st_periods, code)

    # Mark limit status
    df = mark_limit_status(df, limit_ratios, code)

    # Convert date back to integer
    df["date"] = df["date"].dt.strftime("%Y%m%d").astype(int)

    # Select and order columns
    # 包含原始 OHLC（用于涨跌停判断）和 HFQ OHLC（用于技术指标计算）
    cols = [
        "code", "date", "open", "high", "low", "close", 
        "open_hfq", "high_hfq", "low_hfq", "close_hfq",
        "volume", "amount",
        "pre_close", "simple_return", "log_return",
        "is_st", "is_limit_up", "is_limit_down", "is_censored",
    ]
    df = df[cols]

    return df
