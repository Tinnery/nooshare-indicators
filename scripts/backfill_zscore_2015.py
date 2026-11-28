#!/usr/bin/env python3
"""Backfill rolling z-score indicators into indicators.daily_ts_zscore.

Indicators written:
  - volume_zscore_{5,10,20}          (from raw volume)
  - close_hfq_rolling_zscore_{5,10,20} (from HFQ close, research basis)

Behaviour:
  - DRY-RUN by default: computes indicators and reports row counts without
    writing anything. Pass --apply to actually write.
  - Volume Z-score is computed on the calendar-aligned frame: missing
    sessions receive volume=0 and volume_zscore() itself treats zero/non-
    positive volumes as invalid observations, so aligned synthetic rows
    never enter its history window.
  - Close HFQ Z-score follows C2 semantics (recent N valid close
    observations): it is computed directly from the raw Adjusted VIEW
    observation series BEFORE any calendar forward-fill. Synthetic calendar
    rows never enter its window, and only real observation dates can receive
    a value. Invalid prices (non-finite, non-positive) are excluded before
    computation; this is a price-validity rule, not a suspension fact.
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
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so the script runs without installation.
sys.path.insert(0, str(Path(__file__).parent.parent))

from nooshare_indicators.config import MARKET_FACTOR_SET_ID
from nooshare_indicators.infrastructure import database as db
from nooshare_indicators.infrastructure.data_loader import align_with_calendar
from nooshare_indicators.indicators.price_position import close_hfq_rolling_zscore
from nooshare_indicators.indicators.volume import volume_zscore

FACTOR_SET_ID = MARKET_FACTOR_SET_ID
# Start/end are derived from the adjusted view by default; callers may override.
SIGNAL_START_DATE = None
PRE_WARM_START_DATE = None
PERIODS = [5, 10, 20]
TARGET_INDICATOR_NAMES = tuple(
    [f"volume_zscore_{period}" for period in PERIODS]
    + [f"close_hfq_rolling_zscore_{period}" for period in PERIODS]
)
CLOSE_INDICATOR_NAMES = tuple(
    f"close_hfq_rolling_zscore_{period}" for period in PERIODS
)

# Build identity for rows this script writes. Only series actually recomputed
# by this run receive metadata; untouched/legacy rows keep NULL provenance.
ALGORITHM_VERSION_CLOSE = "c2-effective-window-v1"

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
    """Get stock identities that exist in the fixed adjusted view."""
    query = """
        SELECT DISTINCT cs.symbol, cs.exchange, cs.code, cs.list_date, cs.delist_date
        FROM public.canonical_symbol cs
        WHERE cs.asset_class = 'stock'
          AND EXISTS (
              SELECT 1
              FROM public.market_daily_adjusted_bar a
              WHERE a.symbol = cs.symbol
                AND a.factor_set_id = %s::uuid
          )
        ORDER BY cs.exchange, cs.code
    """
    return db.read_sql(query, (FACTOR_SET_ID,))


def get_source_range() -> tuple[int, int]:
    """Return actual min/max date available in the fixed adjusted view."""
    df = db.read_sql(
        """
        SELECT MIN(date) AS min_date, MAX(date) AS max_date
        FROM public.market_daily_adjusted_bar
        WHERE factor_set_id = %s::uuid
        """,
        (FACTOR_SET_ID,),
    )
    if df.empty or pd.isna(df.iloc[0]["min_date"]):
        raise RuntimeError("market_daily_adjusted_bar has no rows for factor_set_id")
    return int(df.iloc[0]["min_date"]), int(df.iloc[0]["max_date"])


def get_source_class_breakdown() -> pd.DataFrame:
    """Report canonical asset classes present in the adjusted view.

    Only asset_class='stock' is eligible; bstock/index/etf/unknown are excluded
    explicitly and reported rather than silently dropped.
    """
    query = """
        SELECT COALESCE(cs.asset_class, 'unknown') AS asset_class,
               count(DISTINCT a.symbol) AS symbols
        FROM (SELECT DISTINCT symbol FROM public.market_daily_adjusted_bar
              WHERE factor_set_id = %s::uuid) a
        LEFT JOIN (SELECT DISTINCT symbol, asset_class FROM public.canonical_symbol) cs
          ON cs.symbol = a.symbol
        GROUP BY 1
        ORDER BY 2 DESC
    """
    return db.read_sql(query, (FACTOR_SET_ID,))


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
    """Load one stock from the fixed unified Adjusted VIEW."""
    query = """
        SELECT symbol, date,
               open::double precision AS open,
               high::double precision AS high,
               low::double precision AS low,
               close::double precision AS close,
               open_hfq, high_hfq, low_hfq, close_hfq,
               volume::double precision AS volume,
               amount::double precision AS amount
        FROM public.market_daily_adjusted_bar
        WHERE symbol = %s
          AND factor_set_id = %s::uuid
          AND date >= %s
          AND date <= %s
        ORDER BY date
    """
    return db.read_sql(query, (symbol, FACTOR_SET_ID, start_date, end_date))


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


def compute_indicators_for_stock(
    aligned_df: pd.DataFrame,
    raw_close_hfq: pd.Series,
) -> dict[str, pd.Series]:
    """Compute all rolling z-score indicators for one stock.

    Volume Z-score keeps the canonical aligned-calendar contract: missing
    sessions receive volume=0 and volume_zscore() itself treats them as
    invalid observations, so synthetic calendar rows never enter its
    history window.

    Close HFQ Z-score follows C2 semantics (recent N valid close
    observations): it must be computed from the raw observation series
    BEFORE any calendar forward-fill, so synthetic calendar rows never
    enter its window. Callers must pass raw_close_hfq with invalid prices
    (non-finite, non-positive) already marked as NaN.
    """
    results = {}

    # Volume Z-score (ddof=1) on the aligned frame.
    for period in PERIODS:
        results[f"volume_zscore_{period}"] = volume_zscore(aligned_df["volume"], period)

    # Close HFQ Rolling Z-score (ddof=1, C2) on raw observations only.
    for period in PERIODS:
        results[f"close_hfq_rolling_zscore_{period}"] = close_hfq_rolling_zscore(
            raw_close_hfq, period
        )

    return results


def replace_stock_records(
    symbol: str,
    start_date: int,
    end_date: int,
    records: list[tuple],
    indicator_names: list[str] | None = None,
    factor_set_id: str | None = None,
    algorithm_version: str | None = None,
    build_id: str | None = None,
) -> None:
    """Atomically replace selected Z-score series in one stock/date range.

    DELETE and INSERT share one transaction. This is intentionally a replacement,
    not an UPSERT: rows whose new value is NaN are absent from records and must no
    longer survive as stale values produced by an older formula. Series omitted
    from indicator_names are outside the recomputation scope and remain untouched.

    Build identity columns are written only when all three provenance values are
    supplied; otherwise the new rows keep NULL metadata (no fabricated identity).
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    metadata_values = (factor_set_id, algorithm_version, build_id)
    metadata_supplied = any(value is not None for value in metadata_values)
    if metadata_supplied and any(value is None for value in metadata_values):
        raise ValueError(
            "build identity requires factor_set_id, algorithm_version and "
            "build_id together; partial provenance would be misleading"
        )

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
    if metadata_supplied:
        insert_query = """
            INSERT INTO indicators.daily_ts_zscore
                (code, date, indicator_name, value,
                 factor_set_id, algorithm_version, build_id)
            VALUES (%s, %s, %s, %s, %s::uuid, %s, %s::uuid)
        """
    else:
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
                    if metadata_supplied:
                        insert_records = [
                            record + metadata_values for record in records
                        ]
                    else:
                        insert_records = records
                    cur.executemany(insert_query, insert_records)
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
    close_only: bool = False,
    build_id: str | None = None,
) -> dict:
    """Rebuild one stock's target indicator rows within the requested date range.

    Two data paths with different contracts (C2 semantics; do not mix them):

    - Volume Z-score: computed on the calendar-aligned frame (missing sessions
      receive volume=0; volume_zscore() treats zero/non-positive volumes as
      invalid observations). Replacement scope is the aligned calendar span
      inside the signal range. In close_only mode Volume rows are completely
      untouched (no delete, no insert, no metadata change).

    - Close HFQ Z-score (C2): computed directly from the raw observation
      series returned by the fixed Adjusted VIEW, BEFORE any calendar
      forward-fill. Synthetic calendar rows never enter its window, and only
      real observation dates receive a value. Replacement scope starts at the
      signal start so stale synthetic values inside suspension/source gaps are
      cleared, and ends at the stock's last real observation so rows outside
      the recomputed span are never deleted. Close rows receive build identity
      (factor_set_id / algorithm_version / build_id) when build_id is supplied.

    Only the Close path writes provenance: this script recomputes the Close
    definition (C2), while existing Volume rows cannot prove their provenance
    and keep NULL metadata (never fabricate build identity).
    """
    target_names = CLOSE_INDICATOR_NAMES if close_only else TARGET_INDICATOR_NAMES
    raw_df = get_stock_data(symbol, pre_warm_start_date, max_date).reset_index(drop=True)
    volume_records: list[tuple] = []
    close_records: list[tuple] = []
    volume_eligible: list[str] = []
    close_eligible: list[str] = []
    eligible_indicator_names: list[str] = []
    skipped_indicator_names: list[str] = []
    reason = None
    volume_replacement: tuple | None = None
    close_replacement: tuple | None = None

    if raw_df.empty:
        reason = "no_source_data"
        skipped_indicator_names = list(target_names)
    else:
        source_dates = pd.to_numeric(raw_df["date"], errors="coerce")
        has_signal_source = bool(
            source_dates.between(signal_start_date, max_date, inclusive="both").any()
        )
        if not has_signal_source:
            # No source rows in the requested signal range is not evidence that
            # old indicator rows are stale. Preserve the existing range.
            reason = "no_signal_dates"
            skipped_indicator_names = list(target_names)
        elif source_dates.duplicated().any():
            raise ValueError(f"duplicate source dates for {symbol}; refusing replacement")
        else:
            # Snapshot raw observation dates BEFORE align_with_calendar: the
            # alignment helper converts the shared 'date' column to datetime
            # in place, and C2 must keep reporting real observation dates.
            raw_observation_dates = [int(value) for value in raw_df["date"].tolist()]
            calendar_dates = (
                get_trading_dates(pre_warm_start_date, max_date)
                if trading_dates is None else trading_dates
            )
            if not calendar_dates:
                reason = "no_trading_calendar"
                skipped_indicator_names = list(target_names)
            else:
                calendar = pd.DataFrame({
                    "trade_date": pd.to_datetime(
                        [str(int(value)) for value in calendar_dates],
                        format="%Y%m%d",
                    )
                })
                # Volume keeps the normal compute pipeline's alignment contract:
                # missing sessions -> forward-filled prices and volume/amount=0.
                aligned_df = align_with_calendar(raw_df, calendar, symbol)
                aligned_df["date"] = (
                    pd.to_datetime(aligned_df["date"])
                    .dt.strftime("%Y%m%d")
                    .astype(int)
                )
                signal_df = aligned_df[aligned_df["date"].between(
                    signal_start_date, max_date, inclusive="both"
                )].copy()

                if signal_df.empty:
                    reason = "no_signal_dates"
                    skipped_indicator_names = list(target_names)
                else:
                    # Raw close observations: invalid prices (non-finite or
                    # non-positive) are excluded BEFORE z-score computation.
                    # This is a price-validity rule, not a suspension fact.
                    raw_close = pd.to_numeric(raw_df["close_hfq"], errors="coerce")
                    raw_close = raw_close.where(
                        np.isfinite(raw_close) & (raw_close > 0)
                    )

                    volume_values = pd.to_numeric(
                        aligned_df["volume"], errors="coerce"
                    ).to_numpy(dtype=np.float64, na_value=np.nan)
                    valid_volume_count = int(
                        np.count_nonzero(
                            np.isfinite(volume_values) & (volume_values > 0)
                        )
                    )
                    close_valid_count = int(raw_close.notna().sum())

                    # Eligibility is per series, not per maximum period, and the
                    # eligible list keeps the alternating volume/close order for
                    # reporting compatibility. Volume needs W prior valid
                    # observations plus current T. Close-HFQ (C2) needs W valid
                    # raw observations overall.
                    for period in PERIODS:
                        volume_name = f"volume_zscore_{period}"
                        close_name = f"close_hfq_rolling_zscore_{period}"
                        if not close_only and valid_volume_count > period:
                            volume_eligible.append(volume_name)
                            eligible_indicator_names.append(volume_name)
                        elif not close_only:
                            skipped_indicator_names.append(volume_name)
                        if close_valid_count >= period:
                            close_eligible.append(close_name)
                            eligible_indicator_names.append(close_name)
                        else:
                            skipped_indicator_names.append(close_name)

                    if eligible_indicator_names:
                        indicators = compute_indicators_for_stock(aligned_df, raw_close)

                        # Volume records: one per aligned calendar session in the
                        # signal range (missing sessions get volume=0 and produce
                        # no value inside volume_zscore itself).
                        if volume_eligible:
                            aligned_date_to_position = {
                                int(date): position
                                for position, date in enumerate(aligned_df["date"].tolist())
                            }
                            for _, row in signal_df.iterrows():
                                record_date = int(row["date"])
                                position = aligned_date_to_position[record_date]
                                for indicator_name in volume_eligible:
                                    series = indicators[indicator_name]
                                    value = (
                                        series.iloc[position]
                                        if position < len(series)
                                        else np.nan
                                    )
                                    if pd.notna(value):
                                        volume_records.append(
                                            (symbol, record_date, indicator_name, float(value))
                                        )
                            volume_replacement = (
                                symbol,
                                int(signal_df["date"].min()),
                                int(signal_df["date"].max()),
                                volume_records,
                                volume_eligible,
                                None,
                                None,
                                None,
                            )

                        # Close records: only real observation dates inside the
                        # signal range. Suspension/source-gap sessions produce
                        # no value; their stale synthetic values are cleared by
                        # the replacement scope below.
                        if close_eligible:
                            for position, record_date in enumerate(raw_observation_dates):
                                if record_date < signal_start_date:
                                    continue
                                for indicator_name in close_eligible:
                                    series = indicators[indicator_name]
                                    value = series.iloc[position]
                                    if pd.notna(value):
                                        close_records.append(
                                            (symbol, record_date, indicator_name, float(value))
                                        )
                            # Provenance is written only when the caller supplies
                            # a build identity for this run; partial identity
                            # would be misleading and is rejected downstream.
                            if build_id is not None:
                                close_metadata = (
                                    FACTOR_SET_ID,
                                    ALGORITHM_VERSION_CLOSE,
                                    build_id,
                                )
                            else:
                                close_metadata = (None, None, None)
                            close_replacement = (
                                symbol,
                                signal_start_date,
                                max(raw_observation_dates),
                                close_records,
                                close_eligible,
                                close_metadata[0],
                                close_metadata[1],
                                close_metadata[2],
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
            "rows": len(volume_records) + len(close_records),
            "replaced": False,
            "eligible_indicators": eligible_indicator_names,
            "skipped_indicators": skipped_indicator_names,
        }

    replacements = [
        replacement
        for replacement in (volume_replacement, close_replacement)
        if replacement is not None
    ]
    if replacements:
        for replacement in replacements:
            replace_stock_records(*replacement)
        return {
            "symbol": symbol,
            "status": "success",
            "reason": reason,
            "rows": len(volume_records) + len(close_records),
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
    parser.add_argument("--start-date", type=int, default=None,
                        help="Signal start date (default: earliest adjusted view date)")
    parser.add_argument("--max-date", type=int, default=None,
                        help="Max date (default: latest adjusted view date)")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Recompute only the three close_hfq_rolling_zscore series; "
        "Volume rows are left completely untouched.",
    )
    args = parser.parse_args()

    # One build identity per script run; only recomputed Close rows carry it.
    build_id = str(uuid.uuid4())

    prepare_target_table(apply=args.apply)

    source_min_date, source_max_date = get_source_range()
    start_date = args.start_date if args.start_date is not None else source_min_date
    max_date = args.max_date if args.max_date is not None else source_max_date
    pre_warm_start_date = source_min_date
    if start_date < source_min_date or max_date > source_max_date or start_date > max_date:
        raise ValueError(
            f"requested range {start_date}-{max_date} outside adjusted source "
            f"{source_min_date}-{source_max_date}"
        )

    print("=== Backfill Z-score Indicators ===")
    print(f"Factor set id: {FACTOR_SET_ID}")
    print(f"Adjusted source range: {source_min_date}-{source_max_date}")
    print(f"Signal start date: {start_date}")
    print(f"Pre-warm start date: {pre_warm_start_date} (source minimum; no earlier history)")
    print(f"Max date: {max_date}")
    print(f"Mode: {'APPLY (writes)' if args.apply else 'DRY-RUN (read-only, pass --apply to write)'}")
    if args.close_only:
        print("Scope: close-only (Volume rows untouched)")
    print(f"Build id: {build_id}")
    print(f"Close algorithm version: {ALGORITHM_VERSION_CLOSE}")
    breakdown = get_source_class_breakdown()
    print("Adjusted symbols by canonical asset_class (only stock eligible):")
    for _, row in breakdown.iterrows():
        print(f"  {row['asset_class']}: {int(row['symbols'])}")
    print()

    stocks_df = get_pit_stocks()
    supported_symbols = set(stocks_df["symbol"].tolist())
    print(f"Eligible canonical stocks in adjusted view: {len(stocks_df)}")

    if args.symbols:
        symbols = [symbol for symbol in args.symbols if symbol in supported_symbols]
        excluded_symbols = [symbol for symbol in args.symbols if symbol not in supported_symbols]
        if excluded_symbols:
            print(f"Excluded unsupported symbols ({len(excluded_symbols)}): {excluded_symbols[:20]}")
        print(f"Processing {len(symbols)} specific supported symbols")
    else:
        query = """
            SELECT DISTINCT symbol FROM public.market_daily_adjusted_bar
            WHERE factor_set_id = %s::uuid
              AND date >= %s AND date <= %s
        """
        active_df = db.read_sql(query, (FACTOR_SET_ID, start_date, max_date))
        active_symbols = set(active_df["symbol"].tolist())

        # PIT filter: must be listed before signal start
        symbols = []
        for _, row in stocks_df.iterrows():
            sym = row["symbol"]
            if sym not in active_symbols:
                continue
            if pd.notna(row.get("delist_date")):
                if pd.to_datetime(row["delist_date"]) < pd.to_datetime(
                    str(start_date), format="%Y%m%d"
                ):
                    continue
            symbols.append(sym)

        print(f"Stocks with adjusted data in signal range: {len(symbols)}")

    print()

    total_rows = 0
    success_count = 0
    skip_count = 0
    error_count = 0
    error_symbols: list[str] = []
    reason_counts: dict[str, int] = {}
    start_time = time.time()

    # Load the trading calendar once for the whole batch; do not issue one extra
    # calendar query per symbol.
    trading_dates = get_trading_dates(pre_warm_start_date, max_date)

    for i, symbol in enumerate(symbols):
        try:
            result = backfill_stock(
                symbol, start_date, pre_warm_start_date, max_date, args.apply,
                trading_dates=trading_dates,
                close_only=args.close_only,
                build_id=build_id,
            )

            reason = result.get("reason")
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

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
                    f"Errors: {error_count} | Rows: {total_rows:,} | "
                    f"ETA: {eta:.0f}s"
                )

        except Exception as e:
            error_count += 1
            error_symbols.append(symbol)
            print(f"ERROR processing {symbol}: {e}")
            continue

    elapsed = time.time() - start_time
    print()
    print("=== Complete ===")
    print(f"Total stocks: {len(symbols)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    if reason_counts:
        print("Skip reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}")
    if error_symbols:
        print(f"Failed symbols ({len(error_symbols)}): {error_symbols[:20]}")
    print(f"Total rows {'written' if args.apply else '(dry-run)'}: {total_rows:,}")
    print(f"Elapsed: {elapsed:.1f}s")
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
