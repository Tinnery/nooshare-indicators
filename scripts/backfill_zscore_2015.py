#!/usr/bin/env python3
"""Backfill rolling z-score indicators into indicators.daily_ts_zscore.

Indicators written:
  - volume_zscore_{5,10,20}          (from raw volume)
  - close_hfq_rolling_zscore_{5,10,20} (from HFQ close, research basis per ADR-0001)

Behaviour:
  - DRY-RUN by default: computes indicators and reports row counts without
    writing anything. Pass --apply to actually write.
  - All SQL is parameterized; no string interpolation of user input.
  - Database connection comes from the shared nooshare_indicators.config
    (PG_HOST / PG_PORT / PG_USER / PG_PASSWORD / PG_DATABASE / PG_SSLMODE
    environment variables or a project-root .env). No credentials in code.

Target table: indicators.daily_ts_zscore (owned by this repo; DDL lives in
nooshare_indicators/infrastructure/database.py::init_schema and is ensured
on startup below).

Usage:
    cd nooshare-indicators
    uv run python scripts/backfill_zscore_2015.py [--apply]
        [--batch-size 100] [--start-date 20150101] [--max-date YYYYMMDD]
        [--symbols sh600519 ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so the script runs without installation.
sys.path.insert(0, str(Path(__file__).parent.parent))

from nooshare_indicators.infrastructure import database as db
from nooshare_indicators.indicators.price_position import close_hfq_rolling_zscore
from nooshare_indicators.indicators.volume import volume_zscore

SIGNAL_START_DATE = 20150101  # Only write signals from this date onwards
PRE_WARM_START_DATE = 20080101  # Load data from this date for pre-warming
PERIODS = [5, 10, 20]

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS indicators.daily_ts_zscore (
    code TEXT NOT NULL,
    date INTEGER NOT NULL,
    indicator_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    PRIMARY KEY (code, date, indicator_name)
)
"""


def ensure_table() -> None:
    db.execute_sql(ENSURE_TABLE_SQL)
    db.execute_sql(
        "CREATE INDEX IF NOT EXISTS idx_daily_ts_zscore_date "
        "ON indicators.daily_ts_zscore (date)"
    )


def get_pit_stocks() -> pd.DataFrame:
    """Get all stocks with list/delist dates for PIT filtering."""
    query = """
        SELECT symbol, list_date, delist_date
        FROM public.canonical_symbol
        WHERE asset_class = 'stock' OR asset_class IS NULL
    """
    return db.read_sql(query)


def get_trading_dates(start_date: int, end_date: int) -> list[int]:
    """Get trading dates in range."""
    query = """
        SELECT TO_CHAR(cal_date, 'YYYYMMDD')::int AS date
        FROM public.trading_calendar
        WHERE cal_date >= TO_DATE(%s, 'YYYYMMDD')
          AND cal_date <= TO_DATE(%s, 'YYYYMMDD')
          AND is_open = true
        ORDER BY cal_date
    """
    df = db.read_sql(query, (str(start_date), str(end_date)))
    return df["date"].tolist()


def get_stock_data(symbol: str, start_date: int, end_date: int) -> pd.DataFrame:
    """Load HFQ daily data for one stock (parameterized; HFQ basis per ADR-0001)."""
    query = """
        SELECT symbol, date, open, high, low, close,
               open_hfq, high_hfq, low_hfq, close_hfq,
               volume, amount
        FROM public.hfq_daily
        WHERE symbol = %s
          AND date >= %s
          AND date <= %s
        ORDER BY date
    """
    return db.read_sql(query, (symbol, start_date, end_date))


def filter_stocks_by_date(stocks_df: pd.DataFrame, target_date: int) -> list[str]:
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


def write_records(records: list[tuple]) -> None:
    """Upsert records into indicators.daily_ts_zscore (parameterized)."""
    insert_query = """
        INSERT INTO indicators.daily_ts_zscore (code, date, indicator_name, value)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (code, date, indicator_name) DO UPDATE SET value = EXCLUDED.value
    """
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_query, records)
        conn.commit()


def backfill_stock(
    symbol: str,
    signal_start_date: int,
    pre_warm_start_date: int,
    max_date: int,
    apply: bool,
) -> dict:
    """Backfill indicators for a single stock."""
    # Load data with pre-warm
    df = get_stock_data(symbol, pre_warm_start_date, max_date)

    if len(df) < max(PERIODS) + 1:
        return {"symbol": symbol, "status": "skipped", "reason": "insufficient_data", "rows": 0}

    # Compute indicators
    indicators = compute_indicators_for_stock(df)

    # Filter to signal dates only
    signal_df = df[df["date"] >= signal_start_date].copy()

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

    if not apply:
        return {"symbol": symbol, "status": "dry_run", "rows": len(records)}

    if records:
        write_records(records)

    return {"symbol": symbol, "status": "success", "rows": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill rolling z-score indicators")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write to the database. WITHOUT this flag the script is a read-only dry-run.",
    )
    parser.add_argument("--start-date", type=int, default=SIGNAL_START_DATE, help="Signal start date")
    parser.add_argument("--max-date", type=int, default=None, help="Max date (default: latest in DB)")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    args = parser.parse_args()

    ensure_table()

    max_date = args.max_date
    if max_date is None:
        df = db.read_sql("SELECT MAX(date) AS max_date FROM public.hfq_daily")
        max_date = int(df["max_date"].iloc[0])

    print("=== Backfill Z-score Indicators ===")
    print(f"Signal start date: {args.start_date}")
    print(f"Pre-warm start date: {PRE_WARM_START_DATE}")
    print(f"Max date: {max_date}")
    print(f"Mode: {'APPLY (writes)' if args.apply else 'DRY-RUN (read-only, pass --apply to write)'}")
    print()

    stocks_df = get_pit_stocks()
    print(f"Total stocks in canonical_symbol: {len(stocks_df)}")

    if args.symbols:
        symbols = args.symbols
        print(f"Processing {len(symbols)} specific symbols")
    else:
        # Symbols that have data in the signal range (parameterized)
        query = """
            SELECT DISTINCT symbol FROM public.hfq_daily
            WHERE date >= %s AND date <= %s
        """
        active_df = db.read_sql(query, (args.start_date, max_date))
        active_symbols = set(active_df["symbol"].tolist())

        # PIT filter: must be listed before signal start
        symbols = []
        for _, row in stocks_df.iterrows():
            sym = row["symbol"]
            if sym not in active_symbols:
                continue
            if pd.notna(row.get("delist_date")):
                if pd.to_datetime(row["delist_date"]) < pd.to_datetime(
                    str(args.start_date), format="%Y%m%d"
                ):
                    continue
            symbols.append(sym)

        print(f"Stocks with data in signal range: {len(symbols)}")

    print()

    total_rows = 0
    success_count = 0
    skip_count = 0
    start_time = time.time()

    for i, symbol in enumerate(symbols):
        try:
            result = backfill_stock(
                symbol, args.start_date, PRE_WARM_START_DATE, max_date, args.apply
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
    print("=== Complete ===")
    print(f"Total stocks: {len(symbols)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total rows {'written' if args.apply else '(dry-run)'}: {total_rows:,}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
