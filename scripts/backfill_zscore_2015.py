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

Target table: indicators.daily_ts_zscore (owned by this repo).
Schema DDL is applied only when --apply is explicitly supplied. A dry-run
checks that the target table exists but never creates or alters database objects.

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
from nooshare_indicators.infrastructure.data_loader import align_with_calendar
from nooshare_indicators.indicators.price_position import close_hfq_rolling_zscore
from nooshare_indicators.indicators.volume import volume_zscore

SIGNAL_START_DATE = 20150101  # Only write signals from this date onwards
PRE_WARM_START_DATE = 20080101  # Load data from this date for pre-warming
PERIODS = [5, 10, 20]
TARGET_INDICATOR_NAMES = tuple(
    [f"volume_zscore_{period}" for period in PERIODS]
    + [f"close_hfq_rolling_zscore_{period}" for period in PERIODS]
)

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


def target_table_exists() -> bool:
    """Check the target relation using SELECT only."""
    df = db.read_sql(
        "SELECT to_regclass('indicators.daily_ts_zscore') AS relation"
    )
    return not df.empty and pd.notna(df.iloc[0]["relation"])


def prepare_target_table(apply: bool) -> None:
    """Apply DDL only in explicit apply mode; dry-run must remain read-only."""
    if apply:
        ensure_table()
        return
    if not target_table_exists():
        raise RuntimeError(
            "indicators.daily_ts_zscore does not exist. Dry-run will not create "
            "tables or indexes; apply the schema explicitly or rerun with --apply."
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


def replace_stock_records(
    symbol: str,
    start_date: int,
    end_date: int,
    records: list[tuple],
    indicator_names: list[str] | None = None,
) -> None:
    """Atomically replace selected Z-score series in one stock/date range.

    DELETE and INSERT share one transaction. This is intentionally a replacement,
    not an UPSERT: rows whose new value is NaN are absent from records and must no
    longer survive as stale values produced by an older formula. Series omitted
    from indicator_names are outside the recomputation scope and remain untouched.
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    target_names = tuple(
        TARGET_INDICATOR_NAMES if indicator_names is None else indicator_names
    )
    if not target_names:
        raise ValueError("at least one indicator name must be selected")
    if len(set(target_names)) != len(target_names):
        raise ValueError("indicator names must not contain duplicates")
    if not set(target_names).issubset(TARGET_INDICATOR_NAMES):
        raise ValueError("replacement includes an unowned indicator name")

    allowed_names = set(target_names)
    for record in records:
        if len(record) != 4:
            raise ValueError("each indicator record must contain code, date, name, value")
        code, record_date, indicator_name, _value = record
        if code != symbol:
            raise ValueError("record symbol does not match replacement scope")
        if not start_date <= int(record_date) <= end_date:
            raise ValueError("record date is outside replacement scope")
        if indicator_name not in allowed_names:
            raise ValueError("record indicator is outside replacement scope")
        if not np.isfinite(float(_value)):
            raise ValueError("record value must be finite")

    delete_query = """
        DELETE FROM indicators.daily_ts_zscore
        WHERE code = %s
          AND date >= %s
          AND date <= %s
          AND indicator_name = ANY(%s)
    """
    insert_query = """
        INSERT INTO indicators.daily_ts_zscore (code, date, indicator_name, value)
        VALUES (%s, %s, %s, %s)
    """

    with db.get_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    delete_query,
                    (symbol, start_date, end_date, list(target_names)),
                )
                if records:
                    cur.executemany(insert_query, records)
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def backfill_stock(
    symbol: str,
    signal_start_date: int,
    pre_warm_start_date: int,
    max_date: int,
    apply: bool,
    trading_dates: list[int] | None = None,
) -> dict:
    """Rebuild one stock's target indicator rows within the requested date range.

    The backfill uses the same trading-calendar alignment as the normal compute
    pipeline. Missing trading sessions are forward-filled for Raw/HFQ prices and
    receive volume=0 before indicators are computed. This is essential because
    close-HFQ rolling scores are intentionally computed on those aligned sessions.
    """
    raw_df = get_stock_data(symbol, pre_warm_start_date, max_date).reset_index(drop=True)
    records = []
    eligible_indicator_names: list[str] = []
    skipped_indicator_names: list[str] = []
    reason = None
    df = raw_df

    if raw_df.empty:
        reason = "no_source_data"
        skipped_indicator_names = list(TARGET_INDICATOR_NAMES)
    else:
        source_dates = pd.to_numeric(raw_df["date"], errors="coerce")
        has_signal_source = bool(
            source_dates.between(signal_start_date, max_date, inclusive="both").any()
        )
        if not has_signal_source:
            # No source rows in the requested signal range is not evidence that
            # old indicator rows are stale. Preserve the existing range.
            reason = "no_signal_dates"
            skipped_indicator_names = list(TARGET_INDICATOR_NAMES)
        elif source_dates.duplicated().any():
            raise ValueError(f"duplicate source dates for {symbol}; refusing replacement")
        else:
            calendar_dates = (
                get_trading_dates(pre_warm_start_date, max_date)
                if trading_dates is None else trading_dates
            )
            if not calendar_dates:
                reason = "no_trading_calendar"
                skipped_indicator_names = list(TARGET_INDICATOR_NAMES)
            else:
                calendar = pd.DataFrame({
                    "trade_date": pd.to_datetime(
                        [str(int(value)) for value in calendar_dates],
                        format="%Y%m%d",
                    )
                })
                # Reuse the normal compute pipeline's exact alignment contract:
                # missing sessions -> forward-filled prices and volume/amount=0.
                df = align_with_calendar(raw_df, calendar, symbol)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)
                signal_df = df[df["date"].between(
                    signal_start_date, max_date, inclusive="both"
                )].copy()

                if signal_df.empty:
                    reason = "no_signal_dates"
                    skipped_indicator_names = list(TARGET_INDICATOR_NAMES)
                else:
                    volume_values = pd.to_numeric(
                        df["volume"], errors="coerce"
                    ).to_numpy(dtype=np.float64, na_value=np.nan)
                    close_values = pd.to_numeric(
                        df["close_hfq"], errors="coerce"
                    ).to_numpy(dtype=np.float64, na_value=np.nan)
                    valid_volume_count = int(
                        np.count_nonzero(np.isfinite(volume_values) & (volume_values > 0))
                    )
                    close_validity = pd.Series(
                        np.isfinite(close_values), dtype=np.int64
                    )

                    # Eligibility is per series, not per maximum period. Volume
                    # needs W prior valid observations plus current T. Close-HFQ
                    # needs one complete W-row window after normal calendar fill.
                    for period in PERIODS:
                        volume_name = f"volume_zscore_{period}"
                        close_name = f"close_hfq_rolling_zscore_{period}"
                        if valid_volume_count > period:
                            eligible_indicator_names.append(volume_name)
                        else:
                            skipped_indicator_names.append(volume_name)

                        has_close_window = bool(
                            close_validity.rolling(
                                window=period, min_periods=period
                            ).sum().ge(period).any()
                        )
                        if has_close_window:
                            eligible_indicator_names.append(close_name)
                        else:
                            skipped_indicator_names.append(close_name)

                    if eligible_indicator_names:
                        indicators = compute_indicators_for_stock(df)
                        date_to_position = {
                            int(date): position
                            for position, date in enumerate(df["date"].tolist())
                        }
                        for _, row in signal_df.iterrows():
                            record_date = int(row["date"])
                            position = date_to_position[record_date]
                            for indicator_name in eligible_indicator_names:
                                series = indicators[indicator_name]
                                value = series.iloc[position] if position < len(series) else np.nan
                                if pd.notna(value):
                                    records.append(
                                        (symbol, record_date, indicator_name, float(value))
                                    )
                        if skipped_indicator_names:
                            reason = "partial_history"
                    else:
                        reason = "insufficient_history"

    if not apply:
        return {
            "symbol": symbol,
            "status": "dry_run",
            "reason": reason,
            "rows": len(records),
            "replaced": False,
            "eligible_indicators": eligible_indicator_names,
            "skipped_indicators": skipped_indicator_names,
        }

    if eligible_indicator_names:
        # Replace only series whose source history is sufficient to recompute
        # them. Empty output for an eligible series is meaningful and clears
        # stale values; missing sessions are present because df was calendar-aligned.
        # Only replace the date interval that was actually materialized by
        # calendar alignment and recomputed. The requested range can extend
        # beyond a symbol's first/last available source session; deleting those
        # unobserved edges would erase rows we did not regenerate.
        replacement_start = int(signal_df["date"].min())
        replacement_end = int(signal_df["date"].max())
        replace_stock_records(
            symbol,
            replacement_start,
            replacement_end,
            records,
            eligible_indicator_names,
        )
        return {
            "symbol": symbol,
            "status": "success",
            "reason": reason,
            "rows": len(records),
            "replaced": True,
            "eligible_indicators": eligible_indicator_names,
            "skipped_indicators": skipped_indicator_names,
        }

    return {
        "symbol": symbol,
        "status": "skipped",
        "reason": reason,
        "rows": 0,
        "replaced": False,
        "eligible_indicators": eligible_indicator_names,
        "skipped_indicators": skipped_indicator_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill rolling z-score indicators")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create required table/index and atomically replace target indicator rows. Without this flag, perform a read-only dry-run.",
    )
    parser.add_argument("--start-date", type=int, default=SIGNAL_START_DATE, help="Signal start date")
    parser.add_argument("--max-date", type=int, default=None, help="Max date (default: latest in DB)")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    args = parser.parse_args()

    prepare_target_table(apply=args.apply)

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

    # Load the trading calendar once for the whole batch; do not issue one extra
    # calendar query per symbol.
    trading_dates = get_trading_dates(PRE_WARM_START_DATE, max_date)

    for i, symbol in enumerate(symbols):
        try:
            result = backfill_stock(
                symbol, args.start_date, PRE_WARM_START_DATE, max_date, args.apply,
                trading_dates=trading_dates,
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
