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
        VWAP series
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
    """Volume Z-Score（已接受研究口径，V0.5 修正）。

    Z(T) = (volume(T) - mean(有效观察 T-W..T-1)) / std(T-W..T-1, ddof=1)

    与旧实现（volume.rolling(period)）的差异：
    - 基准窗口 **不包含 T 日**：T 是被评价的当前观测，不能进入自己的基准；
    - 停牌/无效零成交观察不进入基准窗口（volume <= 0 视为无效）；
    - 仍用 ddof=1 样本标准差；窗口不足（有效观察 < period）返回 NaN；
    - std=0 或 inf 返回 NaN。

    Args:
        volume: 原始成交量序列（不调整复权因子）
        period: 基准窗口长度 W（T 之前的有效观察数）

    Returns:
        Z-score 序列，index 与输入一致。
    """
    out = pd.Series(np.nan, index=volume.index, dtype=float)
    values = volume.to_numpy(dtype=float)
    for i in range(period, len(values)):
        window = values[i - period:i]
        valid = window[window > 0]
        if len(valid) < period:
            continue
        std = valid.std(ddof=1)
        if not np.isfinite(std) or std == 0:
            continue
        current = values[i]
        if not np.isfinite(current) or current <= 0:
            continue  # T 日无效观察不产生信号
        out.iloc[i] = (current - valid.mean()) / std
    return out.replace([np.inf, -np.inf], np.nan)


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
        Turnover Z-Score series
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
