"""The Z-score backfill dry-run must never issue DDL or DML."""

import pandas as pd
import pytest

from scripts import backfill_zscore_2015 as backfill


@pytest.fixture(autouse=True)
def _synthetic_trading_calendar(monkeypatch):
    # Synthetic fixtures use consecutive calendar dates as trading sessions.
    monkeypatch.setattr(
        backfill,
        "get_trading_dates",
        lambda start_date, end_date: [
            int(value.strftime("%Y%m%d"))
            for value in pd.date_range(
                pd.to_datetime(str(start_date), format="%Y%m%d"),
                pd.to_datetime(str(end_date), format="%Y%m%d"),
                freq="D",
            )
        ],
    )


def test_dry_run_checks_target_table_with_select_only(monkeypatch):
    statements = []

    def fake_read_sql(sql, *args, **kwargs):
        statements.append(sql)
        return pd.DataFrame({"relation": ["indicators.daily_ts_zscore"]})

    monkeypatch.setattr(backfill.db, "read_sql", fake_read_sql)
    monkeypatch.setattr(
        backfill,
        "ensure_table",
        lambda: pytest.fail("dry-run must not execute DDL"),
    )

    backfill.prepare_target_table(apply=False)

    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("SELECT ")
    assert "to_regclass" in statements[0]


def test_dry_run_reports_missing_table_without_creating_it(monkeypatch):
    statements = []

    def fake_read_sql(sql, *args, **kwargs):
        statements.append(sql)
        return pd.DataFrame({"relation": [None]})

    monkeypatch.setattr(backfill.db, "read_sql", fake_read_sql)
    monkeypatch.setattr(
        backfill,
        "ensure_table",
        lambda: pytest.fail("dry-run must not execute DDL"),
    )

    with pytest.raises(RuntimeError, match="Dry-run will not create"):
        backfill.prepare_target_table(apply=False)

    assert len(statements) == 1


def test_apply_mode_is_the_only_mode_that_ensures_schema(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "ensure_table", lambda: calls.append("ensure"))
    monkeypatch.setattr(
        backfill.db,
        "read_sql",
        lambda *args, **kwargs: pytest.fail("apply mode does not need a dry-run check"),
    )

    backfill.prepare_target_table(apply=True)

    assert calls == ["ensure"]



class _FakeCursor:
    def __init__(self, calls, fail_insert=False):
        self.calls = calls
        self.fail_insert = fail_insert

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append(("execute", sql, params))

    def executemany(self, sql, records):
        self.calls.append(("executemany", sql, list(records)))
        if self.fail_insert:
            raise RuntimeError("insert failed")


class _FakeConnection:
    def __init__(self, fail_insert=False):
        self.calls = []
        self.commits = 0
        self.rollbacks = 0
        self.cursor_instance = _FakeCursor(self.calls, fail_insert=fail_insert)

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _patch_connection(monkeypatch, connection):
    from contextlib import contextmanager

    @contextmanager
    def fake_get_connection():
        yield connection

    monkeypatch.setattr(backfill.db, "get_connection", fake_get_connection)


def test_replace_stock_records_deletes_scoped_rows_then_inserts_and_commits(monkeypatch):
    connection = _FakeConnection()
    _patch_connection(monkeypatch, connection)
    records = [
        ("sh600519", 20240102, "volume_zscore_5", 2.25),
        ("sh600519", 20240103, "close_hfq_rolling_zscore_20", 1.5),
    ]

    backfill.replace_stock_records("sh600519", 20240101, 20240131, records)

    assert [call[0] for call in connection.calls] == ["execute", "executemany"]
    delete_kind, delete_sql, delete_params = connection.calls[0]
    assert delete_kind == "execute"
    assert "DELETE FROM indicators.daily_ts_zscore" in delete_sql
    assert "code = %s" in delete_sql
    assert "date >= %s" in delete_sql and "date <= %s" in delete_sql
    assert "indicator_name = ANY(%s)" in delete_sql
    assert delete_params == (
        "sh600519", 20240101, 20240131, list(backfill.TARGET_INDICATOR_NAMES)
    )
    assert connection.calls[1][2] == records
    assert connection.commits == 1
    assert connection.rollbacks == 0



def test_replace_stock_records_can_limit_delete_to_selected_series(monkeypatch):
    connection = _FakeConnection()
    _patch_connection(monkeypatch, connection)
    selected = ["volume_zscore_5"]
    records = [("sh600519", 20240102, "volume_zscore_5", 2.25)]

    backfill.replace_stock_records(
        "sh600519", 20240101, 20240131, records, selected
    )

    delete_params = connection.calls[0][2]
    assert delete_params == ("sh600519", 20240101, 20240131, selected)
    assert connection.commits == 1


def test_replace_stock_records_empty_output_still_clears_old_scope(monkeypatch):
    connection = _FakeConnection()
    _patch_connection(monkeypatch, connection)

    backfill.replace_stock_records("sh600519", 20240101, 20240131, [])

    assert len(connection.calls) == 1
    assert connection.calls[0][0] == "execute"
    assert connection.commits == 1
    assert connection.rollbacks == 0


def test_replace_stock_records_rolls_back_delete_when_insert_fails(monkeypatch):
    connection = _FakeConnection(fail_insert=True)
    _patch_connection(monkeypatch, connection)
    records = [("sh600519", 20240102, "volume_zscore_5", 2.25)]

    with pytest.raises(RuntimeError, match="insert failed"):
        backfill.replace_stock_records("sh600519", 20240101, 20240131, records)

    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    "record",
    [
        ("sz000001", 20240102, "volume_zscore_5", 1.0),
        ("sh600519", 20231231, "volume_zscore_5", 1.0),
        ("sh600519", 20240102, "unrelated_indicator", 1.0),
    ],
)
def test_replace_stock_records_rejects_rows_outside_delete_scope(monkeypatch, record):
    connection = _FakeConnection()
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ValueError):
        backfill.replace_stock_records("sh600519", 20240101, 20240131, [record])

    assert connection.calls == []
    assert connection.commits == 0


def _stock_frame(n=31):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = [10.0 + i for i in range(n)]
    return pd.DataFrame({
        "symbol": ["sh600519"] * n,
        "date": [int(value.strftime("%Y%m%d")) for value in dates],
        "open": close,
        "high": [value + 0.1 for value in close],
        "low": [value - 0.1 for value in close],
        "close": close,
        "open_hfq": close,
        "high_hfq": [value + 0.1 for value in close],
        "low_hfq": [value - 0.1 for value in close],
        "close_hfq": close,
        "volume": [100.0 + i for i in range(n)],
        "amount": [10_000.0 + i for i in range(n)],
    })


def _volume_names(names):
    return [name for name in names if name.startswith("volume_")]


def _close_names(names):
    return [name for name in names if name.startswith("close_")]


def _nan_indicator_series(aligned_df, raw_close):
    return {
        name: pd.Series([float("nan")] * len(aligned_df))
        for name in backfill.TARGET_INDICATOR_NAMES
    }


def _reference_close_zscore(valid_values, period):
    """Independent reference for C2 semantics: Z over the most recent
    `period` valid observations including current T, ddof=1.

    Deliberately a plain-Python implementation, not the production function.
    """
    result = []
    for index in range(len(valid_values)):
        if index + 1 < period:
            result.append(None)
            continue
        window = valid_values[index + 1 - period : index + 1]
        mean = sum(window) / period
        variance = sum((value - mean) ** 2 for value in window) / (period - 1)
        std = variance ** 0.5
        result.append(None if std == 0 else (valid_values[index] - mean) / std)
    return result


def test_apply_backfill_replaces_range_even_when_every_new_value_is_nan(monkeypatch):
    frame = _stock_frame()
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240110, 20080101, 20240131, apply=True
    )

    assert result["status"] == "success"
    assert result["rows"] == 0
    assert result["replaced"] is True
    assert len(replacements) == 2
    # Volume is replaced over the aligned calendar span inside the signal range,
    # without build identity (volume provenance cannot be proven in this run).
    assert replacements[0][:3] == ("sh600519", 20240110, 20240131)
    assert replacements[0][4] == _volume_names(backfill.TARGET_INDICATOR_NAMES)
    assert replacements[0][5:] == (None, None, None)
    # Close is replaced from the signal start to the last real observation.
    assert replacements[1][:3] == ("sh600519", 20240110, 20240131)
    assert replacements[1][4] == _close_names(backfill.TARGET_INDICATOR_NAMES)
    # Without an explicit build_id the close rows keep NULL metadata too.
    assert replacements[1][5:] == (None, None, None)


def test_dry_run_backfill_never_replaces_existing_rows(monkeypatch):
    frame = _stock_frame()
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: pytest.fail("dry-run must never DELETE or INSERT"),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240110, 20080101, 20240131, apply=False
    )

    assert result["status"] == "dry_run"
    assert result["rows"] == 0
    assert result["replaced"] is False


@pytest.mark.parametrize("row_count", range(1, 22))
def test_apply_backfill_replaces_only_series_with_sufficient_history(
    monkeypatch, row_count
):
    frame = _stock_frame(n=row_count)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240131, apply=True
    )

    expected = []
    for period in backfill.PERIODS:
        # Volume Z-score needs W prior observations plus current T.
        if row_count >= period + 1:
            expected.append(f"volume_zscore_{period}")
        # Close HFQ rolling Z-score includes T in its W-observation window.
        if row_count >= period:
            expected.append(f"close_hfq_rolling_zscore_{period}")

    assert result["eligible_indicators"] == expected
    assert set(result["skipped_indicators"]) == (
        set(backfill.TARGET_INDICATOR_NAMES) - set(expected)
    )

    last_date = int(frame["date"].max())
    expected_replacements = []
    volume_names = _volume_names(expected)
    close_names = _close_names(expected)
    if volume_names:
        expected_replacements.append(
            ("sh600519", 20240101, last_date, [], volume_names, None, None, None)
        )
    if close_names:
        expected_replacements.append(
            ("sh600519", 20240101, last_date, [], close_names, None, None, None)
        )

    if expected:
        assert result["status"] == "success"
        assert result["replaced"] is True
        assert replacements == expected_replacements
    else:
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_history"
        assert result["replaced"] is False
        assert replacements == []


def test_volume_history_insufficiency_does_not_clear_volume_rows(monkeypatch):
    frame = _stock_frame(n=21)
    frame["volume"] = 0.0
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240131, apply=True
    )

    assert result["eligible_indicators"] == [
        "close_hfq_rolling_zscore_5",
        "close_hfq_rolling_zscore_10",
        "close_hfq_rolling_zscore_20",
    ]
    # Only the close series is inside the replacement scope.
    assert len(replacements) == 1
    assert replacements[0][4] == result["eligible_indicators"]


def test_close_history_insufficiency_does_not_clear_close_rows(monkeypatch):
    frame = _stock_frame(n=21)
    frame["close_hfq"] = float("nan")
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240131, apply=True
    )

    assert result["eligible_indicators"] == [
        "volume_zscore_5",
        "volume_zscore_10",
        "volume_zscore_20",
    ]
    # Only the volume series is inside the replacement scope.
    assert len(replacements) == 1
    assert replacements[0][4] == result["eligible_indicators"]


def test_close_history_missing_values_exclude_nan_observations(monkeypatch):
    """C2 counts valid raw observations; NaN rows are not forward-filled first."""
    frame = _stock_frame(n=21)
    frame["close_hfq"] = [
        value if index % 2 == 0 else float("nan")
        for index, value in enumerate(frame["close_hfq"])
    ]
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240131, apply=True
    )

    # 11 valid close observations: enough for periods 5 and 10, not for 20.
    # Volume keeps all 21 valid observations.
    assert result["eligible_indicators"] == [
        "volume_zscore_5",
        "close_hfq_rolling_zscore_5",
        "volume_zscore_10",
        "close_hfq_rolling_zscore_10",
        "volume_zscore_20",
    ]
    assert len(replacements) == 2
    assert replacements[0][4] == [
        "volume_zscore_5", "volume_zscore_10", "volume_zscore_20",
    ]
    assert replacements[1][4] == [
        "close_hfq_rolling_zscore_5", "close_hfq_rolling_zscore_10",
    ]


def test_missing_source_session_does_not_generate_close_zscore(monkeypatch):
    """A calendar day without a source observation produces no close value."""
    frame = _stock_frame(n=10)
    frame = frame[frame["date"] != 20240106].reset_index(drop=True)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240106, 20240101, 20240110, apply=True
    )

    assert result["status"] == "success"
    assert result["replaced"] is True
    assert len(replacements) == 2
    volume_records = replacements[0][3]
    close_records = replacements[1][3]
    # 2024-01-06 has no source row: no close z-score may be generated there.
    assert not any(record[1] == 20240106 for record in close_records)
    # Volume keeps the aligned-calendar contract but volume=0 on the gap day
    # is an invalid observation, so no volume record either.
    assert not any(
        record[1] == 20240106 and record[2].startswith("volume_zscore_")
        for record in volume_records
    )
    # Close values only exist on real observation dates in the signal range.
    assert sorted(record[1] for record in close_records) == [
        20240107, 20240108, 20240109, 20240110,
    ]


def test_apply_backfill_deletes_only_dates_inside_recomputed_calendar_span(monkeypatch):
    frame = _stock_frame(n=10)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20231220, 20080101, 20240131, apply=True
    )

    assert result["replaced"] is True
    # Volume delete scope stays inside the aligned calendar span.
    assert replacements[0][:3] == ("sh600519", 20240101, 20240110)
    # Close delete scope starts at the signal start (clears stale synthetic
    # values) and ends at the last real observation (never deletes beyond it).
    assert replacements[1][:3] == ("sh600519", 20231220, 20240110)


def test_no_source_rows_in_signal_range_never_clears_existing_rows(monkeypatch):
    frame = _stock_frame(n=21)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240201, 20080101, 20240131, apply=True
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_signal_dates"
    assert result["replaced"] is False
    assert replacements == []


def test_resume_after_source_gap_uses_recent_valid_observations(monkeypatch):
    """After a multi-day source gap, C2 resumes on the most recent valid
    observations and the value matches an independent reference."""
    frame = _stock_frame(n=20)
    frame = frame[
        ~frame["date"].isin([20240111, 20240112, 20240113])
    ].reset_index(drop=True)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240111, 20240101, 20240120, apply=True
    )

    assert result["status"] == "success"
    assert len(replacements) == 2
    close_records = replacements[1][3]

    # No close value inside the gap days, even though the aligned calendar
    # contains synthetic rows for them.
    assert not any(
        record[1] in (20240111, 20240112, 20240113) for record in close_records
    )

    # Resume day 2024-01-14 uses the most recent 5 valid observations
    # (2024-01-07 .. 2024-01-10 and 2024-01-14 itself).
    resume_records = [
        record for record in close_records
        if record[1] == 20240114 and record[2] == "close_hfq_rolling_zscore_5"
    ]
    assert len(resume_records) == 1
    assert resume_records[0][2] == "close_hfq_rolling_zscore_5"

    # Independent reference: valid observation series is 17 entries long.
    valid_values = [10.0 + i for i in range(10)] + [10.0 + i for i in range(13, 20)]
    reference = _reference_close_zscore(valid_values, 5)
    # 2024-01-14 is the 11th valid observation (index 10).
    expected = reference[10]
    assert expected is not None
    assert resume_records[0][3] == pytest.approx(expected, rel=1e-12)


def test_non_positive_and_non_finite_close_prices_are_excluded(monkeypatch):
    """Non-finite and non-positive prices are invalid observations, not
    synthetic values; they never enter the window nor produce a value."""
    frame = _stock_frame(n=10)
    frame.loc[frame["date"] == 20240105, "close_hfq"] = 0.0
    frame.loc[frame["date"] == 20240107, "close_hfq"] = -5.0
    frame.loc[frame["date"] == 20240109, "close_hfq"] = float("inf")
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240110, apply=True
    )

    assert result["status"] == "success"
    assert len(replacements) == 2
    close_records = replacements[1][3]

    # Invalid-price days produce no close record.
    assert not any(
        record[1] in (20240105, 20240107, 20240109) for record in close_records
    )

    # 2024-01-10 window: most recent 5 valid observations
    # (2024-01-03, 04, 06, 08, 10).
    last_records = [
        record for record in close_records if record[1] == 20240110
    ]
    assert len(last_records) == 1

    valid_values = [10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 19.0]
    reference = _reference_close_zscore(valid_values, 5)
    expected = reference[6]
    assert expected is not None
    assert last_records[0][3] == pytest.approx(expected, rel=1e-12)


def test_current_day_without_valid_close_produces_no_close_record(monkeypatch):
    frame = _stock_frame(n=10)
    frame.loc[frame["date"] == 20240110, "close_hfq"] = float("nan")
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240110, apply=True
    )

    assert result["status"] == "success"
    assert len(replacements) == 2
    volume_records = replacements[0][3]
    close_records = replacements[1][3]

    # The last day has no valid close: no close value on it.
    assert not any(record[1] == 20240110 for record in close_records)
    # Volume is unaffected: it still has a value on the last day.
    assert any(
        record[1] == 20240110 and record[2] == "volume_zscore_5"
        for record in volume_records
    )


def test_duplicate_source_dates_rejected(monkeypatch):
    frame = _stock_frame(n=10)
    duplicate = frame[frame["date"] == 20240105].copy()
    frame = pd.concat([frame, duplicate], ignore_index=True)
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: pytest.fail("must reject before any replacement"),
    )

    with pytest.raises(ValueError, match="duplicate source dates"):
        backfill.backfill_stock(
            "sh600519", 20240101, 20080101, 20240110, apply=True
        )

def test_replace_stock_records_writes_build_identity(monkeypatch):
    connection = _FakeConnection()
    _patch_connection(monkeypatch, connection)
    records = [("sh600519", 20240102, "close_hfq_rolling_zscore_5", 1.5)]

    backfill.replace_stock_records(
        "sh600519", 20240101, 20240131, records,
        ["close_hfq_rolling_zscore_5"],
        factor_set_id="11111111-1111-4111-8111-111111111111",
        algorithm_version="c2-effective-window-v1",
        build_id="0f0e0d0c-0b0a-4009-8007-000605040302",
    )

    delete_params = connection.calls[0][2]
    assert delete_params[3] == ["close_hfq_rolling_zscore_5"]
    insert_sql = connection.calls[1][1]
    assert "factor_set_id" in insert_sql
    assert "algorithm_version" in insert_sql
    assert "build_id" in insert_sql
    assert connection.calls[1][2] == [(
        "sh600519", 20240102, "close_hfq_rolling_zscore_5", 1.5,
        "11111111-1111-4111-8111-111111111111",
        "c2-effective-window-v1",
        "0f0e0d0c-0b0a-4009-8007-000605040302",
    )]


def test_replace_stock_records_rejects_partial_metadata(monkeypatch):
    connection = _FakeConnection()
    _patch_connection(monkeypatch, connection)
    records = [("sh600519", 20240102, "close_hfq_rolling_zscore_5", 1.5)]

    with pytest.raises(ValueError, match="partial provenance"):
        backfill.replace_stock_records(
            "sh600519", 20240101, 20240131, records,
            factor_set_id="11111111-1111-4111-8111-111111111111",
            build_id="0f0e0d0c-0b0a-4009-8007-000605040302",
        )

    assert connection.calls == []


def test_close_only_leaves_volume_rows_untouched(monkeypatch):
    """--close-only scope: no volume eligibility, no volume replacement."""
    frame = _stock_frame(n=21)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240131, apply=True, close_only=True
    )

    assert result["eligible_indicators"] == [
        "close_hfq_rolling_zscore_5",
        "close_hfq_rolling_zscore_10",
        "close_hfq_rolling_zscore_20",
    ]
    assert result["skipped_indicators"] == []
    assert len(replacements) == 1
    assert replacements[0][4] == result["eligible_indicators"]


def test_backfill_stock_with_build_id_writes_close_provenance(monkeypatch):
    frame = _stock_frame(n=21)
    replacements = []
    monkeypatch.setattr(backfill, "get_stock_data", lambda *args: frame)
    monkeypatch.setattr(backfill, "compute_indicators_for_stock", _nan_indicator_series)
    monkeypatch.setattr(
        backfill,
        "replace_stock_records",
        lambda *args: replacements.append(args),
    )

    result = backfill.backfill_stock(
        "sh600519", 20240101, 20080101, 20240131, apply=True,
        build_id="0f0e0d0c-0b0a-4009-8007-000605040302",
    )

    assert len(replacements) == 2
    # Volume keeps NULL provenance: its definition is not recomputed here.
    assert replacements[0][5:] == (None, None, None)
    # Close carries the full build identity of this run.
    assert replacements[1][5] == "11111111-1111-4111-8111-111111111111"
    assert replacements[1][6] == "c2-effective-window-v1"
    assert replacements[1][7] == "0f0e0d0c-0b0a-4009-8007-000605040302"
