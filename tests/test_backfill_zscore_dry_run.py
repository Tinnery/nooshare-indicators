"""The Z-score backfill dry-run must never issue DDL or DML."""

import pandas as pd
import pytest

from scripts import backfill_zscore_2015 as backfill


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
