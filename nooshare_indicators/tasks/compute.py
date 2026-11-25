"""Main computation engine: orchestrate all indicator calculations."""
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm

from .. import config
from ..infrastructure import database as db
from ..infrastructure.data_loader import (
    load_trading_calendar,
    load_stock_list,
    load_st_status,
    load_limit_ratio,
    prepare_base_data,
)
from ..indicators import returns as ret
from ..indicators import volatility as vol
from ..indicators import momentum as mom
from ..indicators import volume as vlm
from ..indicators import price_position as pp
from ..indicators import cross_section as cs


def compute_indicators_for_stock(
    code: str,
    calendar: pd.DataFrame,
    st_periods: pd.DataFrame,
    limit_ratios: pd.DataFrame,
    start_date: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all indicators for a single stock.

    Args:
        code: Stock code
        calendar: Trading calendar
        st_periods: ST status periods
        limit_ratios: Limit ratio data
        start_date: Start date (YYYYMMDD)

    Returns:
        Dict with keys: base, volatility, momentum, volume, price_position
    """
    # Load and prepare base data
    df = prepare_base_data(code, calendar, st_periods, limit_ratios, start_date)

    if df.empty:
        return {}

    # Extract series
    # 使用 HFQ 价格进行技术指标计算（消除除权除息跳变）
    # 原始价格保留在 df 中，用于涨跌停判断等需要真实价格的场景
    open_s = df["open_hfq"] if "open_hfq" in df.columns else df["open"]
    high_s = df["high_hfq"] if "high_hfq" in df.columns else df["high"]
    low_s = df["low_hfq"] if "low_hfq" in df.columns else df["low"]
    close_s = df["close_hfq"] if "close_hfq" in df.columns else df["close"]
    volume_s = df["volume"]
    log_return_s = df["log_return"]

    results = {}

    # Base data
    results["base"] = df

    # Volatility indicators
    vol_results = vol.compute_all_volatilities(
        open_s, high_s, low_s, close_s, log_return_s, config.VOLATILITY_PERIODS
    )
    # vol_results 是一个长格式 DataFrame，每行对应一个 (period, date) 组合
    # 需要按 period 和 date 展开
    vol_df = vol_results.copy()
    vol_df["code"] = code
    vol_df["date"] = df["date"].tolist() * len(config.VOLATILITY_PERIODS)
    results["volatility"] = vol_df

    # Momentum indicators
    mom_results = mom.compute_all_momentum(close_s, config.MOMENTUM_PERIODS)
    mom_rows = []
    for name, series in mom_results.items():
        for i, value in enumerate(series):
            mom_rows.append({
                "code": code,
                "date": df["date"].iloc[i],
                "indicator_name": name,
                "value": value,
            })
    results["momentum"] = pd.DataFrame(mom_rows)

    # Volume indicators
    vlm_results = vlm.compute_all_volume(
        open_s, high_s, low_s, close_s, volume_s, config.VOLUME_PERIODS
    )
    vlm_rows = []
    for name, series in vlm_results.items():
        for i, value in enumerate(series):
            vlm_rows.append({
                "code": code,
                "date": df["date"].iloc[i],
                "indicator_name": name,
                "value": value,
            })
    results["volume"] = pd.DataFrame(vlm_rows)

    # Price position indicators
    pp_results = pp.compute_all_price_position(
        open_s, high_s, low_s, close_s,
        config.PRICE_POSITION_PERIODS,
        config.BOLL_PERIOD,
        config.BOLL_STD,
        config.ATR_PERIOD,
    )
    pp_rows = []
    for name, series in pp_results.items():
        for i, value in enumerate(series):
            pp_rows.append({
                "code": code,
                "date": df["date"].iloc[i],
                "indicator_name": name,
                "value": value,
            })
    results["price_position"] = pd.DataFrame(pp_rows)

    return results


def compute_all_indicators(
    codes: Optional[list[str]] = None,
    start_date: Optional[int] = None,
    max_stocks: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all indicators for all stocks (optimized batch mode).

    Args:
        codes: List of stock codes (None = all active stocks)
        start_date: Start date (YYYYMMDD)
        max_stocks: Maximum number of stocks to process (for testing)

    Returns:
        Dict with keys: base, volatility, momentum, volume, price_position
    """
    print("Loading reference data...")
    calendar = load_trading_calendar()
    stock_list = load_stock_list()
    st_periods = load_st_status()
    limit_ratios = load_limit_ratio()

    if codes is None:
        codes = stock_list["stock_code"].tolist()

    if max_stocks:
        codes = codes[:max_stocks]

    print(f"Processing {len(codes)} stocks (batch mode)...")

    # Batch load all stock data in one query
    from ..infrastructure.data_loader import load_all_stocks_daily_batch
    print("Loading all stock data in batch...")
    all_data = load_all_stocks_daily_batch(codes, start_date)
    
    # Aggregate results
    all_base = []
    all_volatility = []
    all_momentum = []
    all_volume = []
    all_price_position = []
    all_zscore = []

    # Process each stock
    for code in tqdm(codes, desc="Computing indicators"):
        try:
            # Filter data for this stock
            stock_data = all_data[all_data["code"] == code].copy()
            
            if stock_data.empty:
                continue

            # Align with calendar
            from ..infrastructure.data_loader import align_with_calendar, compute_pre_close, compute_returns, mark_st_status, mark_limit_status
            stock_data = align_with_calendar(stock_data, calendar, code)
            stock_data = compute_pre_close(stock_data)
            stock_data = compute_returns(stock_data)
            stock_data = mark_st_status(stock_data, st_periods, code)
            stock_data = mark_limit_status(stock_data, limit_ratios, code)
            
            # Convert date back to integer
            if pd.api.types.is_datetime64_any_dtype(stock_data["date"]):
                stock_data["date"] = stock_data["date"].dt.strftime("%Y%m%d").astype(int)
            elif stock_data["date"].dtype == 'object':
                # Already in string format, convert to int
                stock_data["date"] = stock_data["date"].astype(str).str.replace('-', '').astype(int)
            
            # Select columns - include HFQ fields for technical indicators
            cols = [
                "code", "date", 
                "open", "high", "low", "close",  # Raw for limit detection
                "open_hfq", "high_hfq", "low_hfq", "close_hfq",  # HFQ for indicators
                "volume", "amount",
                "pre_close", "simple_return", "log_return",
                "is_st", "is_limit_up", "is_limit_down", "is_censored",
            ]
            df = stock_data[cols]
            
            if df.empty:
                continue

            # Extract series - use HFQ for technical indicators
            # Fallback to Raw only if HFQ is missing (should not happen with current data)
            open_s = df["open_hfq"] if "open_hfq" in df.columns else df["open"]
            high_s = df["high_hfq"] if "high_hfq" in df.columns else df["high"]
            low_s = df["low_hfq"] if "low_hfq" in df.columns else df["low"]
            close_s = df["close_hfq"] if "close_hfq" in df.columns else df["close"]
            volume_s = df["volume"]
            log_return_s = df["log_return"]

            results = {}
            results["base"] = df

            # Volatility indicators
            vol_results = vol.compute_all_volatilities(
                open_s, high_s, low_s, close_s, log_return_s, config.VOLATILITY_PERIODS
            )
            vol_df = vol_results.copy()
            vol_df["code"] = code
            vol_df["date"] = df["date"].tolist() * len(config.VOLATILITY_PERIODS)
            results["volatility"] = vol_df

            # Momentum indicators
            mom_results = mom.compute_all_momentum(close_s, config.MOMENTUM_PERIODS)
            mom_rows = []
            for name, series in mom_results.items():
                for i, value in enumerate(series):
                    mom_rows.append({
                        "code": code,
                        "date": df["date"].iloc[i],
                        "indicator_name": name,
                        "value": value,
                    })
            results["momentum"] = pd.DataFrame(mom_rows)

            # Volume indicators
            vlm_results = vlm.compute_all_volume(
                open_s, high_s, low_s, close_s, volume_s, config.VOLUME_PERIODS
            )
            vlm_rows = []
            for name, series in vlm_results.items():
                for i, value in enumerate(series):
                    vlm_rows.append({
                        "code": code,
                        "date": df["date"].iloc[i],
                        "indicator_name": name,
                        "value": value,
                    })
            results["volume"] = pd.DataFrame(vlm_rows)

            # Price position indicators
            pp_results = pp.compute_all_price_position(
                open_s, high_s, low_s, close_s,
                config.PRICE_POSITION_PERIODS,
                config.BOLL_PERIOD,
                config.BOLL_STD,
                config.ATR_PERIOD,
            )
            pp_rows = []
            for name, series in pp_results.items():
                for i, value in enumerate(series):
                    pp_rows.append({
                        "code": code,
                        "date": df["date"].iloc[i],
                        "indicator_name": name,
                        "value": value,
                    })
            results["price_position"] = pd.DataFrame(pp_rows)

            all_base.append(results["base"])
            all_volatility.append(results["volatility"])
            all_momentum.append(results["momentum"])
            all_volume.append(results["volume"])
            all_price_position.append(results["price_position"])
            # Rolling Z-score for close_hfq (T02)
            zscore_rows = []
            for period in [5, 10, 20]:
                zs = pp.close_hfq_rolling_zscore(close_s, period)
                for i, value in enumerate(zs):
                    if pd.notna(value):
                        zscore_rows.append({
                            "code": code,
                            "date": df["date"].iloc[i],
                            "indicator_name": f"close_hfq_rolling_zscore_{period}",
                            "value": value,
                        })
            if zscore_rows:
                all_zscore.append(pd.DataFrame(zscore_rows))


        except Exception as e:
            print(f"Error processing {code}: {e}")
            continue

    print("Concatenating results...")

    return {
        "base": pd.concat(all_base, ignore_index=True) if all_base else pd.DataFrame(),
        "volatility": pd.concat(all_volatility, ignore_index=True) if all_volatility else pd.DataFrame(),
        "momentum": pd.concat(all_momentum, ignore_index=True) if all_momentum else pd.DataFrame(),
        "volume": pd.concat(all_volume, ignore_index=True) if all_volume else pd.DataFrame(),
        "price_position": pd.concat(all_price_position, ignore_index=True) if all_price_position else pd.DataFrame(),
        "zscore": pd.concat(all_zscore, ignore_index=True) if all_zscore else pd.DataFrame(),
    }


def add_cross_section_to_results(
    results: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Add cross-section rank and z-score to all indicator results.

    Args:
        results: Dict of DataFrames from compute_all_indicators

    Returns:
        Updated results with cross-section metrics
    """
    print("Computing cross-section metrics...")

    # Volatility: rank within each (date, period) — periods are NOT comparable,
    # so ranking must be grouped by period, not mixed across periods.
    if not results["volatility"].empty:
        vol_df = results["volatility"].copy()
        for col in ["vol_ccc", "vol_parkinson", "vol_gk", "vol_rs"]:
            rank_col = f"{col}_rank"
            vol_df[rank_col] = (
                vol_df.groupby(["date", "period"])[col]
                .rank(pct=True, ascending=True)
            )
        results["volatility"] = vol_df

    # Momentum, Volume, Price Position: rank + zscore within each
    # (date, indicator_name) cross-section. Uses vectorized groupby.transform
    # (C-level) instead of a per-group Python callback for speed.
    for key in ["momentum", "volume", "price_position"]:
        df = results[key]
        if df.empty:
            continue

        grp = df.groupby(["date", "indicator_name"])["value"]
        df["cross_section_rank"] = grp.rank(pct=True, ascending=True)
        gmean = grp.transform("mean")
        gstd = grp.transform("std")  # ddof=1 (sample) by default
        zs = (df["value"] - gmean) / gstd
        # Zero std or NaN std → NaN (not 0)
        zs = zs.where(gstd != 0, np.nan)
        zs = zs.where(gstd.notna(), np.nan)
        df["cross_section_zscore"] = zs.where(df["value"].notna(), np.nan)

        results[key] = df

    return results


def save_results_to_db(results: dict[str, pd.DataFrame]):
    """
    Save all results to PostgreSQL.

    Each table is written via chunked COPY (see ``database.copy_from_dataframe``),
    every chunk committing independently. A batch is therefore resumable: if the
    process dies mid-batch, already-written chunks stay durable and a re-run
    overwrites them idempotently; the batch's checkpoint is only set once all
    five tables succeed.

    Args:
        results: Dict of DataFrames from compute_all_indicators
    """
    print("Saving to database...")
    for key, table in [
        ("base", "daily_base_aligned"),
        ("volatility", "daily_volatility"),
        ("momentum", "daily_momentum"),
        ("volume", "daily_volume_indicators"),
        ("price_position", "daily_price_position"),
        ("zscore", "daily_ts_zscore"),
    ]:
        if not results[key].empty:
            print(f"  Saving {len(results[key])} {key} records...")
            db.copy_from_dataframe(results[key], table)
    print("Done!")


def compute_cross_section_for_all_dates(
    start_date: Optional[int] = None,
) -> None:
    """Phase B: Compute cross-section statistics on the full universe.
    
    This function reads all raw indicator values from the database and computes
    cross-section rank and z-score for each (date, indicator_name) group across
    the complete universe of stocks. This ensures that cross-section statistics
    are computed on the full universe, not within batches.
    
    The results are written back to the database, updating the cross_section_rank
    and cross_section_zscore columns in the indicator tables.
    
    Args:
        start_date: Optional start date filter (YYYYMMDD)
    """
    print("Loading raw indicators from database...")
    
    # Load all momentum, volume, and price_position data
    tables = ["daily_momentum", "daily_volume_indicators", "daily_price_position"]
    
    for table in tables:
        print(f"\nProcessing {table}...")
        
        # Build query with optional date filter
        if start_date:
            sql = f"""
            SELECT code, date, indicator_name, value
            FROM indicators.{table}
            WHERE date >= {start_date}
            ORDER BY date, indicator_name, code
            """
        else:
            sql = f"""
            SELECT code, date, indicator_name, value
            FROM indicators.{table}
            ORDER BY date, indicator_name, code
            """
        
        df = db.read_sql(sql)
        
        if df.empty:
            print(f"  No data in {table}")
            continue
        
        print(f"  Loaded {len(df)} rows")
        
        # Compute cross-section statistics
        # Group by (date, indicator_name) and compute rank + zscore
        grp = df.groupby(["date", "indicator_name"])["value"]
        
        # Rank (percentile, ascending)
        df["cross_section_rank"] = grp.rank(pct=True, ascending=True)
        
        # Z-score with ddof=1 (sample std)
        gmean = grp.transform("mean")
        gstd = grp.transform("std")  # pandas default is ddof=1
        
        # Handle zero std and NaN std → NaN
        zs = (df["value"] - gmean) / gstd
        zs = zs.where(gstd != 0, np.nan)
        zs = zs.where(gstd.notna(), np.nan)
        df["cross_section_zscore"] = zs.where(df["value"].notna(), np.nan)
        
        # Replace inf/-inf with NaN
        df["cross_section_zscore"] = df["cross_section_zscore"].replace(
            [np.inf, -np.inf], np.nan
        )
        
        print(f"  Computed cross-section stats for {df['date'].nunique()} dates, "
              f"{df['indicator_name'].nunique()} indicators")
        
        # Write back to database
        # We need to update the existing rows with the new cross-section values
        # Use the same table name but only update the cross-section columns
        update_df = df[["code", "date", "indicator_name", "cross_section_rank", "cross_section_zscore"]]
        
        print(f"  Updating database...")
        db.update_cross_section_stats(update_df, table)
        
        print(f"  Done")
    
    print("\nPhase B complete: All cross-section statistics updated")

