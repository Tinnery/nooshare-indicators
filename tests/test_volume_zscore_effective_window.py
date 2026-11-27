"""Volume Z-score 有效窗口手算 Fixture（V0.7）。

验证 volume_zscore() 的正确性：
- T 日不参与自己的基准统计
- 基准窗口是 T 之前最近 W 个有效历史观察值
- 零值/缺失值/非有限值作为无效观察（不推断原因）
- 向前搜索直到找到 W 个有效观察值
"""
import pytest
import pandas as pd
import numpy as np
from nooshare_indicators.indicators.volume import volume_zscore


class TestVolumeZscoreEffectiveWindow:
    """VZ-01 到 VZ-10 手算 Fixture"""
    
    def test_vz01_basic_case(self):
        """VZ-01: [10,20,30,40], W=3 → 最后 Z=2.0
        
        计算 T=40 时：
        - 历史窗口 = [10, 20, 30]
        - mean = 20, std = 10 (ddof=1)
        - Z = (40 - 20) / 10 = 2.0
        """
        volume = pd.Series([10, 20, 30, 40])
        z = volume_zscore(volume, period=3)
        
        assert pd.isna(z.iloc[0])  # 历史不足
        assert pd.isna(z.iloc[1])  # 历史不足
        assert pd.isna(z.iloc[2])  # 历史不足
        assert abs(z.iloc[3] - 2.0) < 1e-10
    
    def test_vz02_zero_in_middle(self):
        """VZ-02: [10,0,20,30,40], W=3 → T=40 时基准为 [10,20,30]，Z=2.0
        
        计算 T=40 时：
        - 历史有效观察 = [10, 20, 30]（跳过中间的 0）
        - mean = 20, std = 10
        - Z = (40 - 20) / 10 = 2.0
        """
        volume = pd.Series([10, 0, 20, 30, 40])
        z = volume_zscore(volume, period=3)
        
        assert pd.isna(z.iloc[0])
        assert pd.isna(z.iloc[1])  # T=0 无效
        assert pd.isna(z.iloc[2])  # 历史只有 [10]
        assert pd.isna(z.iloc[3])  # 历史只有 [10, 20]
        assert abs(z.iloc[4] - 2.0) < 1e-10
    
    def test_vz03_zero_near_end(self):
        """VZ-03: [10,20,0,30,40], W=3 → T=40 时基准为 [10,20,30]，Z=2.0
        
        计算 T=40 时：
        - 历史有效观察 = [10, 20, 30]（跳过 index=2 的 0）
        - mean = 20, std = 10
        - Z = 2.0
        """
        volume = pd.Series([10, 20, 0, 30, 40])
        z = volume_zscore(volume, period=3)
        
        assert pd.isna(z.iloc[0])
        assert pd.isna(z.iloc[1])
        assert pd.isna(z.iloc[2])  # T=0 无效
        assert pd.isna(z.iloc[3])  # 历史只有 [10, 20]
        assert abs(z.iloc[4] - 2.0) < 1e-10
    
    def test_vz04_zero_before_current(self):
        """VZ-04: [10,20,30,0,40], W=3 → T=40 时基准为 [10,20,30]，Z=2.0
        
        计算 T=40 时：
        - 历史有效观察 = [10, 20, 30]（跳过 index=3 的 0）
        - mean = 20, std = 10
        - Z = 2.0
        """
        volume = pd.Series([10, 20, 30, 0, 40])
        z = volume_zscore(volume, period=3)
        
        assert pd.isna(z.iloc[0])
        assert pd.isna(z.iloc[1])
        assert pd.isna(z.iloc[2])
        assert pd.isna(z.iloc[3])  # T=0 无效
        assert abs(z.iloc[4] - 2.0) < 1e-10
    
    def test_vz05_insufficient_history(self):
        """VZ-05: 历史有效观察不足 W → NaN"""
        volume = pd.Series([10, 20])
        z = volume_zscore(volume, period=3)
        
        assert all(pd.isna(z))
    
    def test_vz06_zero_std(self):
        """VZ-06: 历史标准差为 0 → NaN"""
        volume = pd.Series([10, 10, 10, 20])
        z = volume_zscore(volume, period=3)
        
        # 历史 [10, 10, 10] std=0 → NaN
        assert pd.isna(z.iloc[3])
    
    def test_vz07_zero_current(self):
        """VZ-07: 当前成交量为 0 → NaN"""
        volume = pd.Series([10, 20, 30, 0])
        z = volume_zscore(volume, period=3)
        
        # T=0 无效 → NaN
        assert pd.isna(z.iloc[3])
    
    def test_vz08_nan_current(self):
        """VZ-08: 当前成交量为 NaN → NaN"""
        volume = pd.Series([10, 20, 30, np.nan])
        z = volume_zscore(volume, period=3)
        
        assert pd.isna(z.iloc[3])
    
    def test_vz09_no_lookahead_future_change(self):
        """VZ-09: 修改未来成交量不改变之前的 Z-score"""
        volume1 = pd.Series([10, 20, 30, 40, 50])
        volume2 = pd.Series([10, 20, 30, 40, 999])
        
        z1 = volume_zscore(volume1, period=3)
        z2 = volume_zscore(volume2, period=3)
        
        # T=3 的 Z-score 不应受 T=4 影响
        assert abs(z1.iloc[3] - z2.iloc[3]) < 1e-10
    
    def test_vz10_current_change_affects_future(self):
        """VZ-10: 修改当前成交量只影响当前及后续正确窗口的值"""
        volume1 = pd.Series([10, 20, 30, 40, 50])
        volume2 = pd.Series([10, 20, 30, 999, 50])
        
        z1 = volume_zscore(volume1, period=3)
        z2 = volume_zscore(volume2, period=3)
        
        # T=3 的 Z-score 应该不同
        assert abs(z1.iloc[3] - z2.iloc[3]) > 1.0
        
        # T=4 的 Z-score 也应该不同（因为窗口包含 T=3）
        assert abs(z1.iloc[4] - z2.iloc[4]) > 1.0


class TestVolumeZscoreEdgeCases:
    """边界情况测试"""
    
    def test_inf_value(self):
        """无穷值处理"""
        volume = pd.Series([10, 20, 30, np.inf])
        z = volume_zscore(volume, period=3)
        
        assert pd.isna(z.iloc[3])
    
    def test_consecutive_zero_observations(self):
        """连续零值观察；不推断原因"""
        volume = pd.Series([10, 0, 0, 0, 40])
        z = volume_zscore(volume, period=3)
        
        # 只有 T=0 和 T=4 有效，历史不足 3 个
        assert pd.isna(z.iloc[4])
    
    def test_valid_observation_after_long_zero_gap(self):
        """长时间零值间隔后恢复有效观察"""
        volume = pd.Series([10, 20, 30, 0, 0, 0, 0, 0, 40])
        z = volume_zscore(volume, period=3)
        
        # T=40 时历史有效观察 = [10, 20, 30]
        assert abs(z.iloc[8] - 2.0) < 1e-10
    
    def test_isolation_between_stocks(self):
        """不同股票之间的计算隔离"""
        # 模拟两只股票
        vol_a = pd.Series([10, 20, 30, 40])
        vol_b = pd.Series([100, 200, 300, 400])
        
        z_a = volume_zscore(vol_a, period=3)
        z_b = volume_zscore(vol_b, period=3)
        
        # 两只股票的 Z-score 应该相同（相对变化一致）
        assert abs(z_a.iloc[3] - z_b.iloc[3]) < 1e-10
    
    def test_different_periods(self):
        """不同 period 的独立性"""
        volume = pd.Series([10, 20, 30, 40, 50, 60])
        
        z3 = volume_zscore(volume, period=3)
        z5 = volume_zscore(volume, period=5)
        
        # T=4 时：
        # period=3: 历史 [20, 30, 40], mean=30, std=10, Z=(50-30)/10=2.0
        # period=5: 历史不足 5 个 → NaN
        assert abs(z3.iloc[4] - 2.0) < 1e-10
        assert pd.isna(z5.iloc[4])
    
    def test_all_zeros(self):
        """全零序列"""
        volume = pd.Series([0, 0, 0, 0])
        z = volume_zscore(volume, period=3)
        
        assert all(pd.isna(z))
    
    def test_mixed_invalid_values(self):
        """混合无效值"""
        volume = pd.Series([10, np.nan, 20, 0, 30, 40])
        z = volume_zscore(volume, period=3)
        
        # T=40 时历史有效观察 = [10, 20, 30]
        assert abs(z.iloc[5] - 2.0) < 1e-10
    
    def test_exact_period_boundary(self):
        """恰好达到 period 边界"""
        volume = pd.Series([10, 20, 30, 40])
        z = volume_zscore(volume, period=3)
        
        # T=3 时历史恰好 3 个 [10, 20, 30]
        assert not pd.isna(z.iloc[3])
    
    def test_negative_volume_treated_as_invalid(self):
        """负成交量视为无效"""
        volume = pd.Series([10, 20, 30, -5, 40])
        z = volume_zscore(volume, period=3)
        
        # T=40 时历史有效观察 = [10, 20, 30]（跳过 -5）
        assert abs(z.iloc[4] - 2.0) < 1e-10


class TestVolumeZscoreNumericalStability:
    """数值稳定性测试"""
    
    def test_large_values(self):
        """大数值"""
        volume = pd.Series([1e9, 2e9, 3e9, 4e9])
        z = volume_zscore(volume, period=3)
        
        # 应该正常计算，不溢出
        assert not pd.isna(z.iloc[3])
        assert abs(z.iloc[3] - 2.0) < 1e-6
    
    def test_small_values(self):
        """小数值"""
        volume = pd.Series([1e-6, 2e-6, 3e-6, 4e-6])
        z = volume_zscore(volume, period=3)
        
        assert not pd.isna(z.iloc[3])
        assert abs(z.iloc[3] - 2.0) < 1e-6
    
    def test_very_close_values(self):
        """非常接近的值（测试 std 精度）"""
        volume = pd.Series([10.0, 10.0 + 1e-10, 10.0 + 2e-10, 10.0 + 3e-10])
        z = volume_zscore(volume, period=3)
        
        # std 非常小但不为零，应该能计算
        assert not pd.isna(z.iloc[3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
