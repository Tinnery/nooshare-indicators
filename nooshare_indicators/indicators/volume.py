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
    """Volume Z-Score（T-1 有效观察窗口）。

    Z(T) = (V_T - mean(V_{t-W:t-1})) / std(V_{t-W:t-1}, ddof=1)

    规则：
    - 当前 T 不参与自己的基准窗口；
    - 基准是 T 之前最近 W 个有效成交量观察值，而不是最近 W 个日历位置；
    - 只有有限且大于 0 的成交量才是有效观察；0、负数、NaN 和 inf 不进入窗口；
    - 当前值无效、历史有效观察不足 W 个或样本标准差为 0 时返回 NaN；
    - 使用 ddof=1 样本标准差；
    - 仅按输入序列的位置计算，不排序、不跨股票拼接。

    上游日历对齐会把部分缺失日成交量填成 0。本函数将 0 作为无效观察，
    但不会擅自判定其原因究竟是停牌还是源数据缺口。

    Args:
        volume: 单只股票、按交易日期排列的原始成交量序列（不做价格复权）
        period: 基准窗口长度 W（T 之前的有效观察数）

    Returns:
        与输入 index/name 完全一致的 Z-score Series。
    """
    numeric = pd.to_numeric(volume, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    valid_positions = np.flatnonzero(np.isfinite(values) & (values > 0))

    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(valid_positions) > period:
        # 压缩成有效观察序列后 shift(1)：每个被评分值只使用此前 W 个有效观察。
        valid_values = values[valid_positions]
        prior = pd.Series(valid_values).shift(1)
        rolling_mean = prior.rolling(window=period, min_periods=period).mean().to_numpy()
        rolling_std = prior.rolling(window=period, min_periods=period).std(ddof=1).to_numpy()

        valid_std = np.isfinite(rolling_mean) & np.isfinite(rolling_std) & (rolling_std > 0)
        scores = np.full(len(valid_values), np.nan, dtype=np.float64)
        np.divide(
            valid_values - rolling_mean,
            rolling_std,
            out=scores,
            where=valid_std,
        )
        result[valid_positions] = scores

    return pd.Series(result, index=volume.index, name=volume.name)

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
