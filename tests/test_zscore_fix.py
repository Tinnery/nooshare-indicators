"""Tests for Z-score calculation correctness fixes.

Covers:
- Rolling Z-score: ddof=1, min_periods, zero std, inf check
- Cross-sectional Z-score: ddof=1, zero std, NaN handling, batch independence
- HFQ input usage in compute paths
- PIT stock pool filtering
"""
import numpy as np
import pandas as pd
import pytest


# ── Rolling Z-score ──────────────────────────────────────────────────

class TestRollingZscore:
    """Volume zscore: T 之前 W 个有效观察为基准，ddof=1，T 不进入窗口。"""

    def _make_series(self, values):
        return pd.Series(values, dtype=float)

    def test_ddof1_matches_hand_calculation(self):
        """index4(40): 基准=[10,20,30] mean=20 std=10 -> z=(40-20)/10=2.0"""
        from nooshare_indicators.indicators.volume import volume_zscore
        vol = self._make_series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = volume_zscore(vol, period=3)
        assert abs(result.iloc[3] - 2.0) < 1e-10
        assert abs(result.iloc[4] - 2.0) < 1e-10  # [20,30,40] mean=30, T=50, z=(50-30)/10

    def test_window_insufficient_returns_nan(self):
        from nooshare_indicators.indicators.volume import volume_zscore
        vol = self._make_series([100.0, 200.0, 300.0])
        assert volume_zscore(vol, period=5).isna().all()

    def test_current_day_not_in_window(self):
        """V0.5 修正：T 日不进入基准（旧实现的 test_current_day_included 已作废）。"""
        from nooshare_indicators.indicators.volume import volume_zscore
        vol = self._make_series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = volume_zscore(vol, period=3)
        # index3: 基准=[10,20,30], z=2.0；若含T则 z=1.0
        assert abs(result.iloc[3] - 2.0) < 1e-10

    def test_no_lookahead(self):
        from nooshare_indicators.indicators.volume import volume_zscore
        base = volume_zscore(self._make_series([10., 20., 30., 40., 50.]), 3)
        changed = volume_zscore(self._make_series([10., 20., 30., 40., 999.]), 3)
        # T=3 的基准=[10,20,30]，不受 index4 改动影响
        assert abs(base.iloc[3] - changed.iloc[3]) < 1e-10
        # index4 本身是被评价的 T 日，改动它会改变 z（这是正确的）
        assert abs(base.iloc[4] - 2.0) < 1e-10
        assert abs(changed.iloc[4] - 96.9) < 1e-10

    def test_zero_std_returns_nan(self):
        from nooshare_indicators.indicators.volume import volume_zscore
        assert volume_zscore(self._make_series([5.]*6), 3).isna().all()

    def test_all_zero_window_returns_nan(self):
        from nooshare_indicators.indicators.volume import volume_zscore
        assert volume_zscore(self._make_series([0.]*6), 3).isna().all()

    def test_suspended_zero_excluded_from_window(self):
        """停牌零成交不进入基准窗口。"""
        from nooshare_indicators.indicators.volume import volume_zscore
        vol = self._make_series([10., 0., 30., 40., 50., 60.])
        result = volume_zscore(vol, period=3)
        # index5: 基准=[30,40,50] mean=40 std=10 -> z=(60-40)/10=2.0
        assert abs(result.iloc[5] - 2.0) < 1e-10

    def test_no_inf_in_output(self):
        from nooshare_indicators.indicators.volume import volume_zscore
        result = volume_zscore(self._make_series([1., 1.+1e-15, 1., 1., 1.]), 3)
        assert not np.isinf(result.dropna()).any()

    def test_different_periods(self):
        from nooshare_indicators.indicators.volume import volume_zscore
        vol = self._make_series([10., 20., 30., 40., 50., 60., 70.])
        for p in (2, 3, 4):
            r = volume_zscore(vol, period=p)
            assert r.iloc[:p].isna().all()
            assert r.iloc[p:].notna().any()

# ── Cross-sectional Z-score ──────────────────────────────────────────

class TestCrossSectionalZscore:
    """Test cross_section_zscore() contract."""

    def test_ddof1_matches_hand_calculation(self):
        """Cross-section std must use ddof=1."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        values = pd.Series([10.0, 20.0, 30.0])
        # mean=20, std(ddof=1)=10
        # zscores: [-1, 0, 1]
        result = cross_section_zscore(values)
        expected = pd.Series([-1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_single_sample_returns_nan(self):
        """N < 2: std undefined, must return NaN."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        values = pd.Series([42.0])
        result = cross_section_zscore(values)
        assert result.isna().all()

    def test_zero_std_returns_nan(self):
        """All-equal values: std=0, must return NaN, not 0."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        values = pd.Series([5.0, 5.0, 5.0])
        result = cross_section_zscore(values)
        assert result.isna().all()

    def test_nan_not_counted_in_stats(self):
        """NaN values must not participate in mean/std calculation."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        values = pd.Series([10.0, np.nan, 20.0, 30.0])
        result = cross_section_zscore(values)

        # Valid values: [10, 20, 30], mean=20, std(ddof=1)=10
        # zscores for valid: [-1, NaN, 0, 1]
        assert abs(result.iloc[0] - (-1.0)) < 1e-10
        assert np.isnan(result.iloc[1])
        assert abs(result.iloc[2] - 0.0) < 1e-10
        assert abs(result.iloc[3] - 1.0) < 1e-10

    def test_empty_series(self):
        """Empty input returns empty output."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        values = pd.Series([], dtype=float)
        result = cross_section_zscore(values)
        assert result.empty

    def test_unordered_input_same_result(self):
        """Input order must not affect z-score values (same index → same result)."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        values_a = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
        values_b = pd.Series([30.0, 10.0, 20.0], index=[0, 1, 2])

        result_a = cross_section_zscore(values_a)
        result_b = cross_section_zscore(values_b)

        # Same index, same values → same result
        # values_a: [10,20,30] → z=[-1, 0, 1]
        # values_b: [30,10,20] → z=[1, -1, 0]
        assert abs(result_a.iloc[0] - (-1.0)) < 1e-10
        assert abs(result_b.iloc[0] - 1.0) < 1e-10


# ── Cross-sectional stats ────────────────────────────────────────────

class TestCrossSectionStats:
    """Test compute_cross_section_stats() contract."""

    def test_ddof1(self):
        """Stats std must use ddof=1."""
        from nooshare_indicators.indicators.cross_section import compute_cross_section_stats

        values = pd.Series([10.0, 20.0, 30.0])
        stats = compute_cross_section_stats(values)

        assert stats["count"] == 3
        assert abs(stats["mean"] - 20.0) < 1e-10
        # std(ddof=1) of [10,20,30] = 10
        assert abs(stats["std"] - 10.0) < 1e-10

    def test_empty_returns_nan_stats(self):
        from nooshare_indicators.indicators.cross_section import compute_cross_section_stats

        values = pd.Series([], dtype=float)
        stats = compute_cross_section_stats(values)
        assert stats["count"] == 0
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])


# ── Batch independence ───────────────────────────────────────────────

class TestBatchIndependence:
    """Cross-section results must be independent of batch size."""

    def test_cross_section_batch_independence(self):
        """Simulate batch processing: same final result regardless of batch splits."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        # Simulate 401 stocks with known values
        np.random.seed(42)
        n_stocks = 401
        values = pd.Series(np.random.randn(n_stocks) * 10 + 50)

        # Full cross-section z-score (ground truth)
        expected = cross_section_zscore(values)

        # Simulate batch processing: split into batches of 200, compute raw values,
        # then compute cross-section on the combined values
        batch_size = 200
        batches = []
        for i in range(0, n_stocks, batch_size):
            batches.append(values.iloc[i:i + batch_size])

        # Recombine (Phase B logic)
        combined = pd.concat(batches)
        result = cross_section_zscore(combined)

        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_different_batch_sizes_same_result(self):
        """Batch size 1, 200, and all-at-once must give same cross-section."""
        from nooshare_indicators.indicators.cross_section import cross_section_zscore

        np.random.seed(123)
        n = 50
        values = pd.Series(np.random.randn(n))

        expected = cross_section_zscore(values)

        for bs in [1, 7, 20, 50, 100]:
            chunks = [values.iloc[i:i + bs] for i in range(0, n, bs)]
            combined = pd.concat(chunks)
            result = cross_section_zscore(combined)
            pd.testing.assert_series_equal(result, expected, atol=1e-10,
                                           check_names=False)


# ── PIT stock pool ───────────────────────────────────────────────────

class TestPITStockPool:
    """Test filter_stocks_by_date() PIT logic."""

    def test_filter_by_date(self):
        """Stocks must be filtered by list_date and delist_date."""
        from nooshare_indicators.infrastructure.data_loader import filter_stocks_by_date

        # Use the column names that filter_stocks_by_date expects
        stock_list = pd.DataFrame({
            "stock_code": ["A", "B", "C", "D"],
            "list_date": [20200101, 20190101, 20210101, 20180101],
            "delist_date": [None, 20200601, None, 20191201],
        })

        # T = 20200301
        result = filter_stocks_by_date(stock_list, 20200301)
        codes = result  # filter_stocks_by_date returns list[str]

        # A: listed 20200101, not delisted → included
        assert "A" in codes
        # B: listed 20190101, delisted 20200601, T=20200301 <= 20200601 → included
        assert "B" in codes
        # C: listed 20210101, T=20200301 < 20210101 → NOT included
        assert "C" not in codes
        # D: listed 20180101, delisted 20191201, T=20200301 > 20191201 → NOT included
        assert "D" not in codes

    def test_delisted_stock_included_in_its_active_period(self):
        """A delisted stock should be included for dates when it was active."""
        from nooshare_indicators.infrastructure.data_loader import filter_stocks_by_date

        stock_list = pd.DataFrame({
            "stock_code": ["X"],
            "list_date": [20190101],
            "delist_date": [20200601],
        })

        # T = 20200101 (before delist) → included
        result = filter_stocks_by_date(stock_list, 20200101)
        assert "X" in result  # returns list[str]

        # T = 20210101 (after delist) → NOT included
        result = filter_stocks_by_date(stock_list, 20210101)
        assert "X" not in result  # returns list[str]


# ── HFQ input ────────────────────────────────────────────────────────

class TestHFQInput:
    """Verify that compute paths require HFQ fields (ADR-0001)."""

    def test_compute_pre_close_uses_hfq(self):
        """compute_pre_close should use close_hfq when available."""
        from nooshare_indicators.infrastructure.data_loader import compute_pre_close

        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0],
            "close_hfq": [100.0, 110.0, 120.0],
        })
        result = compute_pre_close(df)

        # Should use close_hfq, not close
        assert np.isnan(result["pre_close"].iloc[0])
        assert result["pre_close"].iloc[1] == 100.0
        assert result["pre_close"].iloc[2] == 110.0

    def test_compute_pre_close_raises_without_hfq(self):
        """Missing close_hfq must raise, never fall back to raw close (ADR-0001)."""
        from nooshare_indicators.infrastructure.data_loader import compute_pre_close

        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0],
        })
        with pytest.raises(ValueError, match="close_hfq"):
            compute_pre_close(df)

    def test_compute_returns_uses_hfq(self):
        """compute_returns should use close_hfq when available."""
        from nooshare_indicators.infrastructure.data_loader import compute_returns

        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0],
            "close_hfq": [100.0, 110.0, 120.0],
            "pre_close": [np.nan, 100.0, 110.0],
        })
        result = compute_returns(df)

        # simple_return at index 1: (110 - 100) / 100 = 0.1
        assert abs(result["simple_return"].iloc[1] - 0.1) < 1e-10

    def test_compute_returns_raises_without_hfq(self):
        """Missing close_hfq must raise, never fall back to raw close (ADR-0001)."""
        from nooshare_indicators.infrastructure.data_loader import compute_returns

        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0],
        })
        with pytest.raises(ValueError, match="close_hfq"):
            compute_returns(df)


# ── align_with_calendar HFQ support ─────────────────────────────────

class TestAlignCalendarHFQ:
    """Verify align_with_calendar handles HFQ columns."""

    def test_hfq_columns_forward_filled(self):
        """HFQ price columns should be forward-filled on missing days."""
        from nooshare_indicators.infrastructure.data_loader import align_with_calendar

        # Create stock data with a gap
        df = pd.DataFrame({
            "date": pd.to_datetime(["20240102", "20240104"], format="%Y%m%d"),
            "open": [10.0, 12.0],
            "high": [11.0, 13.0],
            "low": [9.0, 11.0],
            "close": [10.5, 12.5],
            "open_hfq": [100.0, 120.0],
            "high_hfq": [110.0, 130.0],
            "low_hfq": [90.0, 110.0],
            "close_hfq": [105.0, 125.0],
            "volume": [1000, 1200],
            "amount": [10500.0, 15000.0],
        })

        calendar = pd.DataFrame({
            "trade_date": pd.to_datetime(["20240102", "20240103", "20240104"], format="%Y%m%d"),
        })

        result = align_with_calendar(df, calendar, "000001")

        # 20240103 should be forward-filled
        row_0103 = result[result["date"] == pd.Timestamp("20240103")].iloc[0]
        assert row_0103["close"] == 10.5  # forward-filled from 0102
        assert row_0103["close_hfq"] == 105.0  # HFQ also forward-filled
        assert row_0103["volume"] == 0  # missing volume filled with 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
