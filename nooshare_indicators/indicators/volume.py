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
    """
    Volume Z-Score.

    Z = (volume - mean(volume, period)) / std(volume, period)

    Measures how many standard deviations current volume is from the mean.
    
    Uses sample standard deviation (ddof=1) for unbiased estimation.
    Returns NaN when standard deviation is zero or window is insufficient.

    Args:
        volume: Volume series
        period: Rolling window size

    Returns:
        Volume Z-Score series
    """
    rolling_mean = volume.rolling(window=period, min_periods=period).mean()
    rolling_std = volume.rolling(window=period, min_periods=period).std(ddof=1)
    
    # Compute Z-score, handling zero std and invalid values
    zscore = (volume - rolling_mean) / rolling_std
    
    # Replace inf/-inf with NaN (zero std case)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    return zscore


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
