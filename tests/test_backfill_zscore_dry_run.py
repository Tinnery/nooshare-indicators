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


def _nan_indicator_series(frame):
    return {
        name: pd.Series([float("nan")] * len(frame))
        for name in backfill.TARGET_INDICATOR_NAMES
    }


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
    assert len(replacements) == 1
    assert replacements[0] == (
        "sh600519",
        20240110,
        20240131,
        [],
        result["eligible_indicators"],
    )


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
        # Close HFQ rolling Z-score includes T in its W-row window.
        if row_count >= period:
            expected.append(f"close_hfq_rolling_zscore_{period}")

    assert result["eligible_indicators"] == expected
    assert set(result["skipped_indicators"]) == (
        set(backfill.TARGET_INDICATOR_NAMES) - set(expected)
    )
    if expected:
        assert result["status"] == "success"
        assert result["replaced"] is True
        assert replacements == [
            (
                "sh600519",
                20240101,
                20240131,
                [],
                expected,
            )
        ]
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
    assert replacements[0][4] == result["eligible_indicators"]


def test_close_history_with_missing_values_is_forward_filled_before_eligibility(monkeypatch):
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

    # The canonical compute path forward-fills aligned close-HFQ rows, so the
    # backfill must recompute (and replace) those series rather than preserve/drop them.
    assert result["eligible_indicators"] == list(backfill.TARGET_INDICATOR_NAMES)
    assert replacements[0][4] == result["eligible_indicators"]


def test_missing_trading_session_is_recomputed_for_close_hfq_series(monkeypatch):
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
    # 2024-01-06 has no source row: align_with_calendar forward-fills close_hfq
    # from 2024-01-05, so the five-row close rolling z-score is recomputed there.
    gap_rows = [
        record for record in replacements[0][3]
        if record[1] == 20240106
    ]
    assert gap_rows == [
        ("sh600519", 20240106, "close_hfq_rolling_zscore_5", pytest.approx(gap_rows[0][3]))
    ]
    assert len(gap_rows) == 1


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
