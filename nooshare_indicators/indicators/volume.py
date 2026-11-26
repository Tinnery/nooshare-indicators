"""Volume indicators: VWAP, OBV, Volume Ratio, Volume Z-Score."""
import pandas as pd
import numpy as np


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int,
) -> pd.Series:
    """
    Volume Weighted Average Price.

    VWAP = sum(typical_price * volume) / sum(volume)
    where typical_price = (high + low + close) / 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        period: Rolling window size

    Returns:
        VW series
    """
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).rolling(window=period).sum()
    cumulative_vol = volume.rolling(window=period).sum()
    return cumulative_tp_vol / cumulative_vol


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.

    OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]

    Args:
        close: Close prices
        volume: Volume

    Returns:
        OBV series
    """
    direction = np.sign(close.diff())
    obv_values = (direction * volume).cumsum()
    return obv_values


def volume_ratio(volume: pd.Series, period: int = 5) -> pd.Series:
    """
    Volume Ratio (量比).

    Volume Ratio = current_volume / avg_volume(period)

    Measures current trading activity relative to recent average.

    Args:
        volume: Volume series
        period: Rolling window for average (default 5)

    Returns:
        Volume ratio series
    """
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume


def volume_zscore(volume: pd.Series, period: int) -> pd.Series:
    """Volume Z-Score（研究口径，V0.7 修正）。

    Z(T) = (V_T - mean(V_{t-W:t-1})) / std(V_{t-W:t-1}, ddof=1)

    定义：
    - T 是被评价的当前观测，不参与自己的基准统计；
    - 基准窗口是 T 之前最近 W 个**有效**历史成交量观察值；
    - 停牌/零成交/NaN/inf 观察不进入基准窗口；
    - 当前值无效（<=0, NaN, inf）时返回 NaN；
    - 历史有效观察不足 W 个时返回 NaN；
    - 标准差为 0 或 inf 时返回 NaN；
    - 使用 ddof=1 样本标准差。

    与 V0.5 实现的差异：
    - V0.5 只检查固定前 period 个位置，停牌时有效数不足直接跳过；
    - V0.7 向前搜索直到找到 period 个有效历史观察值。

    Args:
        volume: 原始成交量序列（不调整复权因子）
        period: 基准窗口长度 W（T 之前的有效观察数）

    Returns:
        Z-score 序列，index 与输入一致。
    """
    out = pd.Series(np.nan, index=volume.index, dtype=float)
    values = volume.to_numpy(dtype=float)
    
    # 维护最近 period 个有效历史观察值的滑动窗口
    history = []
    
    for i in range(len(values)):
        current = values[i]
        
        # 当前值有效性检查
        current_valid = np.isfinite(current) and current > 0
        
        # 计算 Z-score（使用历史窗口，不含当前值）
        if current_valid and len(history) >= period:
            hist_arr = np.array(history[-period:])
            mean = hist_arr.mean()
            std = hist_arr.std(ddof=1)
            if np.isfinite(std) and std > 0:
                out.iloc[i] = (current - mean) / std
        
        # 当前值有效时加入历史窗口（供后续计算使用）
        if current_valid:
            history.append(current)
    
    return out


def turnover_zscore(
    volume: pd.Series,
    total_shares: float,
    period: int,
) -> pd.Series:
    """
    Turnover Rate Z-Score.

    turnover = volume / total_shares
    zscore = (turnover - mean(turnover)) / std(turnover)

    Args:
        volume: Volume series
        total_shares: Total outstanding shares
        period: Rolling window size

    Returns:
        Turnover Z-score series
    """
    turnover = volume / total_shares
    return volume_zscore(turnover, period)


def compute_all_volume(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    periods: list[int],
) -> dict[str, pd.Series]:
    """
    Compute all volume indicators for multiple periods.

    Args:
        open: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        periods: List of periods

    Returns:
        Dict mapping indicator_name to series
    """
    results = {}

    # VWAP for each period
    for period in periods:
        results[f"vwap_{period}"] = vwap(high, low, close, volume, period)

    # OBV (single series, no period parameter)
    results["obv"] = obv(close, volume)

    # Volume ratio for each period
    for period in periods:
        results[f"volume_ratio_{period}"] = volume_ratio(volume, period)

    # Volume Z-Score for each period
    for period in periods:
        results[f"volume_zscore_{period}"] = volume_zscore(volume, period)

    return results
