#!/usr/bin/env python3
"""
Backfill Rolling Z-score indicators to daily_ts_zscore table.

Target: indicators.daily_ts_zscore
Indicators:
  - volume_zscore_5, volume_zscore_10, volume_zscore_20
  - close_hfq_rolling_zscore_5, close_hfq_rolling_zscore_10, close_hfq_rolling_zscore_20

Date range: 2015-01-01 onwards (with pre-warm from 2008)
Stock pool: PIT (canonical_symbol with list_date/delist_date)
Environment: DEV (nooshare_dev)

Usage:
    cd nooshare-indicators
    python scripts/backfill_zscore_2015.py [--dry-run] [--batch-size 100] [--start-date 20150101]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nooshare_indicators.indicators.volume import volume_zscore
from nooshare_indicators.indicators.price_position import close_hfq_rolling_zscore


# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "nooshare_dev",
    "user": "travis",
}

SIGNAL_START_DATE = 20150101  # Only write signals from this date onwards
PRE_WARM_START_DATE = 20080101  # Load data from this date for pre-warming
PERIODS = [5, 10, 20]


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_pit_stocks(conn) -> pd.DataFrame:
    """Get all stocks with list/delist dates for PIT filtering."""
    query = """
        SELECT symbol, list_date, delist_date
        FROM public.canonical_symbol
        WHERE asset_class = 'stock' OR asset_class IS NULL
    """
    return pd.read_sql(query, conn)


def get_trading_dates(conn, start_date: int, end_date: int) -> list[int]:
    """Get trading dates in range."""
    query = f"""
        SELECT TO_CHAR(cal_date, 'YYYYMMDD')::int as date
        FROM public.trading_calendar
        WHERE cal_date >= '{start_date}' AND cal_date <= '{end_date}'
          AND is_open = true
        ORDER BY cal_date
    """
    df = pd.read_sql(query, conn)
    return df["date"].tolist()


def get_stock_data(conn, symbol: str, start_date: int, end_date: int) -> pd.DataFrame:
    """Load HFQ daily data for a stock."""
    query = f"""
        SELECT symbol, date, open, high, low, close,
               open_hfq, high_hfq, low_hfq, close_hfq,
               volume, amount
        FROM public.hfq_daily
        WHERE symbol = '{symbol}'
          AND date >= {start_date}
          AND date <= {end_date}
        ORDER BY date
    """
    return pd.read_sql(query, conn)


def filter_stocks_by_date(
    stocks_df: pd.DataFrame, target_date: int
) -> list[str]:
    """PIT filter: return symbols valid on target_date."""
    target_dt = pd.to_datetime(str(target_date), format="%Y%m%d")

    mask = pd.Series(True, index=stocks_df.index)

    if "list_date" in stocks_df.columns:
        list_mask = stocks_df["list_date"].isna() | (
            pd.to_datetime(stocks_df["list_date"]) <= target_dt
        )
        mask = mask & list_mask

    if "delist_date" in stocks_df.columns:
        delist_mask = stocks_df["delist_date"].isna() | (
            pd.to_datetime(stocks_df["delist_date"]) >= target_dt
        )
        mask = mask & delist_mask

    return stocks_df.loc[mask, "symbol"].tolist()


def compute_indicators_for_stock(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute all rolling z-score indicators for a stock."""
    results = {}

    # Volume Z-score (ddof=1)
    for period in PERIODS:
        results[f"volume_zscore_{period}"] = volume_zscore(df["volume"], period)

    # Close HFQ Rolling Z-score (ddof=1)
    for period in PERIODS:
        results[f"close_hfq_rolling_zscore_{period}"] = close_hfq_rolling_zscore(
            df["close_hfq"], period
        )

    return results


def backfill_stock(
    conn,
    symbol: str,
    signal_start_date: int,
    pre_warm_start_date: int,
    max_date: int,
    dry_run: bool = False,
) -> dict:
    """Backfill indicators for a single stock."""
    # Load data with pre-warm
    df = get_stock_data(conn, symbol, pre_warm_start_date, max_date)

    if len(df) < max(PERIODS) + 1:
        return {"symbol": symbol, "status": "skipped", "reason": "insufficient_data", "rows": 0}

    # Compute indicators
    indicators = compute_indicators_for_stock(df)

    # Filter to signal dates only
    signal_mask = df["date"] >= signal_start_date
    signal_df = df[signal_mask].copy()

    if len(signal_df) == 0:
        return {"symbol": symbol, "status": "skipped", "reason": "no_signal_dates", "rows": 0}

    # Build records for insertion
    records = []
    for _, row in signal_df.iterrows():
        date = row["date"]
        idx = df.index[df["date"] == date][0]

        for indicator_name, series in indicators.items():
            value = series.iloc[idx] if idx < len(series) else np.nan
            if pd.notna(value):
                records.append((symbol, date, indicator_name, float(value)))

    if dry_run:
        return {"symbol": symbol, "status": "dry_run", "rows": len(records)}

    # Insert into daily_ts_zscore (upsert)
    if records:
        insert_query = """
            INSERT INTO indicators.daily_ts_zscore (code, date, indicator_name, value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (code, date, indicator_name) DO UPDATE SET value = EXCLUDED.value
        """
        cur = conn.cursor()
        cur.executemany(insert_query, records)
        conn.commit()
        cur.close()

    return {"symbol": symbol, "status": "success", "rows": len(records)}


def main():
    parser = argparse.ArgumentParser(description="Backfill Rolling Z-score indicators")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of stocks per batch")
    parser.add_argument("--start-date", type=int, default=SIGNAL_START_DATE, help="Signal start date")
    parser.add_argument("--max-date", type=int, default=None, help="Max date (default: latest in DB)")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    args = parser.parse_args()

    conn = get_conn()

    # Get trading dates
    max_date = args.max_date
    if max_date is None:
        df = pd.read_sql("SELECT MAX(date) as max_date FROM public.hfq_daily", conn)
        max_date = int(df["max_date"].iloc[0])

    print(f"=== Backfill Z-score Indicators ===")
    print(f"Signal start date: {args.start_date}")
    print(f"Pre-warm start date: {PRE_WARM_START_DATE}")
    print(f"Max date: {max_date}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Get PIT stocks
    stocks_df = get_pit_stocks(conn)
    print(f"Total stocks in canonical_symbol: {len(stocks_df)}")

    # Filter to stocks that have data in signal range
    if args.symbols:
        symbols = args.symbols
        print(f"Processing {len(symbols)} specific symbols")
    else:
        # Get symbols that have data after signal_start_date
        query = f"""
            SELECT DISTINCT symbol FROM public.hfq_daily
            WHERE date >= {args.start_date} AND date <= {max_date}
        """
        active_df = pd.read_sql(query, conn)
        active_symbols = set(active_df["symbol"].tolist())

        # Filter by PIT: must be listed before max_date
        symbols = []
        for _, row in stocks_df.iterrows():
            sym = row["symbol"]
            if sym not in active_symbols:
                continue
            # PIT check: listed before max_date
            if pd.notna(row.get("delist_date")):
                if pd.to_datetime(row["delist_date"]) < pd.to_datetime(str(args.start_date), format="%Y%m%d"):
                    continue
            symbols.append(sym)

        print(f"Stocks with data in signal range: {len(symbols)}")

    print()

    # Process stocks
    total_rows = 0
    success_count = 0
    skip_count = 0
    start_time = time.time()

    for i, symbol in enumerate(symbols):
        try:
            result = backfill_stock(
                conn, symbol, args.start_date, PRE_WARM_START_DATE, max_date, args.dry_run
            )

            if result["status"] == "success":
                success_count += 1
                total_rows += result["rows"]
            elif result["status"] == "dry_run":
                total_rows += result["rows"]
            else:
                skip_count += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(symbols) - i - 1) / rate if rate > 0 else 0
                print(
                    f"Progress: {i + 1}/{len(symbols)} "
                    f"({100 * (i + 1) / len(symbols):.1f}%) | "
                    f"Success: {success_count} | Skip: {skip_count} | "
                    f"Rows: {total_rows:,} | "
                    f"ETA: {eta:.0f}s"
                )

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            skip_count += 1
            continue

    elapsed = time.time() - start_time
    print()
    print(f"=== Complete ===")
    print(f"Total stocks: {len(symbols)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total rows written: {total_rows:,}")
    print(f"Elapsed: {elapsed:.1f}s")

    conn.close()


if __name__ == "__main__":
    main()
