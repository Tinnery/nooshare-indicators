"""C2 真实观察语义 + factor_status 传播 回归测试。

覆盖任务书 §6.1 / §4.5 / §4.6：
  - align_with_calendar 标记合成日（is_synthetic），不以 ffill 价格充当真实观察
  - close_hfq_rolling_zscore 排除合成日后滑窗只含真实观察
  - 停牌日不制造额外有效窗口观察
  - 合成日不产出独立 C2 值（值为 NaN）
  - factor_status 经 align_with_calendar 保留，且不被 ffill 造假
  - loader 查询包含 factor_status 列
"""
import numpy as np
import pandas as pd
import pytest

from nooshare_indicators.indicators.price_position import close_hfq_rolling_zscore
from nooshare_indicators.infrastructure import data_loader


# ── C2 真实观察语义 ───────────────────────────────────────────────────

class TestC2RealObservations:
    def test_synthetic_days_excluded_from_window(self):
        """真实观察 10,11,12 + 2 个合成日 → 合成日无独立 C2 值。"""
        close = pd.Series([10.0, 11.0, 11.0, 11.0, 12.0])
        synth = pd.Series([False, False, True, True, False])
        out = close_hfq_rolling_zscore(close, period=3, is_synthetic=synth)
        assert pd.isna(out.iloc[2])
        assert pd.isna(out.iloc[3])

    def test_synthetic_equal_to_exclusion_by_dropna(self):
        """标记合成日的效果等价于把这些点当作缺失。"""
        close = pd.Series([10.0, 11.0, 11.0, 12.0])
        synth = pd.Series([False, False, True, False])
        with_mask = close_hfq_rolling_zscore(close, period=2, is_synthetic=synth)
        ref = close_hfq_rolling_zscore(
            pd.Series([10.0, 11.0, np.nan, 12.0]), period=2)
        assert with_mask.iloc[1] == pytest.approx(ref.iloc[1])
        assert with_mask.iloc[3] == pytest.approx(ref.iloc[3])

    def test_no_synthetic_param_preserves_old_behavior(self):
        """不传 is_synthetic 时行为与改造前一致（向后兼容）。"""
        close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        a = close_hfq_rolling_zscore(close, period=3)
        b = close_hfq_rolling_zscore(close, period=3, is_synthetic=None)
        pd.testing.assert_series_equal(a, b)

    def test_invalid_price_still_excluded(self):
        """Inf / NaN / 0 / 负价 语义不变，仍被排除。"""
        close = pd.Series([10.0, np.inf, np.nan, 0.0, -1.0, 11.0, 12.0])
        synth = pd.Series([False] * 7)
        out = close_hfq_rolling_zscore(close, period=2, is_synthetic=synth)
        assert pd.notna(out.iloc[6])
        assert pd.isna(out.iloc[1]) and pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])

    def test_suspension_does_not_inflate_window(self):
        """停牌复牌：合成日不进窗口，第 N 个真实观察才触发计算。"""
        close = pd.Series([10.0, 11.0, 11.0, 11.0, 11.0, 12.0, 13.0])
        synth = pd.Series([False, False, True, True, True, False, False])
        out = close_hfq_rolling_zscore(close, period=3, is_synthetic=synth)
        assert pd.isna(out.iloc[2])
        assert pd.isna(out.iloc[4])
        assert pd.notna(out.iloc[5])


# ── align_with_calendar 合成标记 ──────────────────────────────────────

def _calendar():
    return pd.DataFrame({"trade_date": pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"])})


def _bar_frame(dates, with_status=False):
    n = len(dates)
    data = {
        "date": pd.to_datetime(dates),
        "open": [10.0 + i for i in range(n)],
        "high": [10.5 + i for i in range(n)],
        "low": [9.5 + i for i in range(n)],
        "close": [10.0 + i for i in range(n)],
        "open_hfq": [100.0 + 10 * i for i in range(n)],
        "high_hfq": [105.0 + 10 * i for i in range(n)],
        "low_hfq": [95.0 + 10 * i for i in range(n)],
        "close_hfq": [100.0 + 10 * i for i in range(n)],
        "volume": [1000 + 100 * i for i in range(n)],
        "amount": [1.0 + 0.1 * i for i in range(n)],
    }
    if with_status:
        data["factor_status"] = ["valid"] + ["incomplete"] * (n - 1)
    return pd.DataFrame(data)


class TestAlignMarksSynthetic:
    def test_gap_day_marked_synthetic(self):
        """日历有、行情无的交易日必须标 is_synthetic=True。"""
        df = _bar_frame(["2026-01-05", "2026-01-06", "2026-01-09"])
        out = data_loader.align_with_calendar(df, _calendar(), code="600000")
        out = out.set_index("date")
        assert out.loc["2026-01-05", "is_synthetic"] == False  # noqa: E712
        assert out.loc["2026-01-07", "is_synthetic"] == True  # noqa: E712
        assert out.loc["2026-01-08", "is_synthetic"] == True  # noqa: E712
        assert out.loc["2026-01-09", "is_synthetic"] == False  # noqa: E712

    def test_synthetic_judged_on_close_hfq(self):
        """is_synthetic 以 close_hfq 为准（C2 实际消费列）。"""
        df = _bar_frame(["2026-01-05", "2026-01-09"])
        out = data_loader.align_with_calendar(df, _calendar(), code="600000")
        assert int(out["is_synthetic"].sum()) == 3  # 1/6,1/7,1/8

    def test_factor_status_not_forward_filled(self):
        """factor_status 不得被 ffill 伪造：合成日保持 NULL。"""
        df = _bar_frame(["2026-01-05", "2026-01-09"], with_status=True)
        out = data_loader.align_with_calendar(df, _calendar(), code="600000")
        out = out.set_index("date")
        assert out.loc["2026-01-05", "factor_status"] == "valid"
        assert out.loc["2026-01-09", "factor_status"] == "incomplete"
        assert pd.isna(out.loc["2026-01-07", "factor_status"])

    def test_volume_fill_contract_unchanged(self):
        """Volume 对齐契约不变：缺失日 volume=0。"""
        df = _bar_frame(["2026-01-05", "2026-01-09"])
        out = data_loader.align_with_calendar(df, _calendar(), code="600000")
        out = out.set_index("date")
        assert out.loc["2026-01-07", "volume"] == 0


# ── loader 读取 factor_status ─────────────────────────────────────────

class TestLoaderSelectsFactorStatus:
    def test_single_loader_query_has_factor_status(self):
        src = open(data_loader.__file__).read()
        assert src.count("volume, amount, factor_status") >= 2

    def test_batch_loader_reads_same_factor_set_id(self):
        """批量与单股必须用同一 config.MARKET_FACTOR_SET_ID。"""
        from nooshare_indicators import config
        src = open(data_loader.__file__).read()
        assert src.count("config.MARKET_FACTOR_SET_ID") >= 2
        assert config.MARKET_FACTOR_SET_ID == "0560b30c-1f8c-4aef-8607-f1751d101164"