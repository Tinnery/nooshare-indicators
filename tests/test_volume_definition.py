"""Volume Z-score 口径差异证据 Fixture。

对照 legacy rolling(含T) 与已接受定义(T前W个有效观察)，
确认现行实现 = 已接受口径。历史已存储指标不回填。
"""
import unittest
import numpy as np
import pandas as pd

from nooshare_indicators.indicators.volume import volume_zscore


def legacy_rolling(volume, period):
    """旧实现：基准窗口包含 T 日（已废弃，仅作对照证据）。"""
    return ((volume - volume.rolling(period, min_periods=period).mean()) /
            volume.rolling(period, min_periods=period).std(ddof=1))


class TestVolumeZscoreDefinition(unittest.TestCase):
    def test_hand_calculation_diff(self):
        vol = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(legacy_rolling(vol, 3).iloc[3], 1.0, places=10)
        self.assertAlmostEqual(volume_zscore(vol, 3).iloc[3], 2.0, places=10)

    def test_zero_observation_excluded(self):
        vol = pd.Series([10.0, 0.0, 30.0, 40.0, 50.0, 60.0])
        self.assertAlmostEqual(volume_zscore(vol, 3).iloc[5], (60.0 - 40.0) / 10.0, places=10)

    def test_implementation_matches_accepted(self):
        vol = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(volume_zscore(vol, 3).iloc[3], 2.0, places=10)


if __name__ == "__main__":
    unittest.main()
