"""Schema-init consistency and input-pollution defense tests.

Covers the round-4 closure requirements:
- backfill apply path creates the same schema as database.init_schema()
  (§5.1: provenance columns present on a brand-new database)
- close_hfq_rolling_zscore() defends against polluted input on its own
  (§4.5: NaN/Inf/zero/negative/non-numeric excluded, duplicate index fails)
"""

import numpy as np
import pandas as pd
import pytest

from nooshare_indicators.indicators.price_position import close_hfq_rolling_zscore


# --------------------------------------------------------------------------
# §4.5 central-function defense
# --------------------------------------------------------------------------

def _to_float(value) -> float:
    """Reference-side coercion: non-numeric values are not valid observations."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _expected_zscore(values: list[float], period: int) -> list[float | None]:
    """Independent reference: manual rolling z-score over valid observations."""
    valid = [
        v for v in (_to_float(item) for item in values)
        if np.isfinite(v) and v > 0
    ]
    out: list[float | None] = []
    for i in range(len(valid)):
        if i + 1 < period:
            out.append(None)
            continue
        window = valid[i + 1 - period : i + 1]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std == 0 or not np.isfinite(std):
            out.append(None)
        else:
            z = (window[-1] - mean) / std
            out.append(z if np.isfinite(z) else None)
    return out


def _assert_matches_reference(values, period):
    idx = pd.RangeIndex(len(values))
    series = pd.Series(values, index=idx, dtype=object)
    result = close_hfq_rolling_zscore(series, period)
    expected = _expected_zscore(values, period)
    valid_positions = [
        i for i, v in enumerate(values)
        if np.isfinite(_to_float(v)) and _to_float(v) > 0
    ]
    for pos_in_valid, original_pos in enumerate(valid_positions):
        exp = expected[pos_in_valid]
        got = result.iloc[original_pos]
        if exp is None:
            assert pd.isna(got), f"pos {original_pos}: expected NaN, got {got}"
        else:
            assert got == pytest.approx(exp, abs=1e-12), (
                f"pos {original_pos}: expected {exp}, got {got}"
            )


def test_close_zscore_excludes_inf_zero_negative_and_text():
    values = [10.0, float("inf"), 0.0, -5.0, 11.0, 12.0, "bad", 13.0,
              float("-inf"), np.nan, 14.0, 15.0, 16.0, 17.0, 18.0]
    # Valid observations: 10, 11, 12, 13, 14, 15, 16, 17, 18 (9 obs)
    result = close_hfq_rolling_zscore(pd.Series(values, dtype=object), 5)
    valid_positions = [0, 4, 5, 7, 10, 11, 12, 13, 14]
    # positions 1,2,3,8,9 are invalid observations -> NaN
    for pos in (1, 2, 3, 8, 9):
        assert pd.isna(result.iloc[pos]), f"polluted position {pos} must be NaN"
    # first 4 valid observations have no full window -> NaN
    assert pd.isna(result.iloc[valid_positions[0]])
    assert pd.isna(result.iloc[valid_positions[1]])
    assert pd.isna(result.iloc[valid_positions[2]])
    assert pd.isna(result.iloc[valid_positions[3]])
    # 5th valid observation onwards: finite value
    assert np.isfinite(result.iloc[valid_positions[4]])
    _assert_matches_reference(values, 5)


def test_close_zscore_duplicate_index_raises():
    series = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        index=[0, 1, 2, 2, 3, 4],
    )
    with pytest.raises(ValueError, match="unique index"):
        close_hfq_rolling_zscore(series, 5)


def test_close_zscore_ddof1_and_current_day_included():
    # Window [10, 20]: mean 15, std(ddof=1) = 7.0710678, z for T=20
    series = pd.Series([10.0, 20.0], index=[0, 1])
    result = close_hfq_rolling_zscore(series, 2)
    expected = (20.0 - 15.0) / np.std([10.0, 20.0], ddof=1)
    assert result.iloc[1] == pytest.approx(expected, abs=1e-12)
    assert pd.isna(result.iloc[0])


def test_close_zscore_zero_std_returns_nan():
    series = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    result = close_hfq_rolling_zscore(series, 5)
    assert result.isna().all()


def test_close_zscore_insufficient_history_all_nan():
    series = pd.Series([10.0, 11.0, 12.0])
    result = close_hfq_rolling_zscore(series, 5)
    assert result.isna().all()
    assert list(result.index) == list(series.index)


def test_close_zscore_matches_independent_reference_random():
    rng = np.random.default_rng(42)
    values = rng.uniform(5, 50, size=200).tolist()
    for bad_pos in (30, 77, 150):
        values[bad_pos] = np.nan
    values[100] = 0.0
    values[101] = -3.0
    values[102] = float("inf")
    for period in (5, 10, 20):
        _assert_matches_reference(values, period)


# --------------------------------------------------------------------------
# §5.1 schema-consistency (string-level contract between the two paths)
# --------------------------------------------------------------------------

def _table_columns_from_ddl(ddl: str) -> set[str]:
    """Extract column names from the CREATE TABLE body (between the parens)."""
    body = ddl[ddl.find("(") + 1 : ddl.rfind(")")]
    columns = set()
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        first = line.split()[0]
        if first.upper() in {"PRIMARY", "UNIQUE", "CONSTRAINT", "FOREIGN", "CHECK"}:
            continue
        columns.add(first)
    return columns


def test_backfill_ensure_table_matches_init_schema_contract():
    """Both apply paths must declare the same column set for daily_ts_zscore."""
    import inspect
    import nooshare_indicators.infrastructure.database as db_mod
    import scripts.backfill_zscore_2015 as bf

    # Extract the daily_ts_zscore CREATE TABLE from the init_schema source
    source = inspect.getsource(db_mod)
    marker = "CREATE TABLE IF NOT EXISTS indicators.daily_ts_zscore"
    start = source.find(marker)
    assert start != -1, "daily_ts_zscore DDL missing from init_schema"
    end = source.find(");", start)
    init_ddl = source[start:end]

    init_columns = _table_columns_from_ddl(init_ddl)
    ensure_columns = _table_columns_from_ddl(bf.ENSURE_TABLE_SQL)

    assert {"factor_set_id", "algorithm_version", "build_id"} <= init_columns, (
        "init_schema must declare provenance columns"
    )
    assert init_columns == ensure_columns, (
        f"schema drift: init_schema={init_columns} vs ensure_table={ensure_columns}"
    )


def test_backfill_migration_sql_is_idempotent_add_column():
    import scripts.backfill_zscore_2015 as bf

    assert "ADD COLUMN IF NOT EXISTS factor_set_id" in bf.ENSURE_MIGRATION_SQL
    assert "ADD COLUMN IF NOT EXISTS algorithm_version" in bf.ENSURE_MIGRATION_SQL
    assert "ADD COLUMN IF NOT EXISTS build_id" in bf.ENSURE_MIGRATION_SQL
    # Migration must not touch the primary key
    assert "PRIMARY KEY" not in bf.ENSURE_MIGRATION_SQL
