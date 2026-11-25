"""Momentum and trend indicators: RSI, MACD, ROC, KAMA, EMA, SMA, DEMA."""
import pandas as pd
import numpy as np
from typing import Optional


def sma(close: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Formula: mean(close, window=period)

    Args:
        close: Close prices
        period: Window size

    Returns:
        SMA series
    """
    return close.rolling(window=period).mean()


def ema(close: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """
    Exponential Moving Average.

    Formula: EMA[t] = alpha * close[t] + (1 - alpha) * EMA[t-1]
    where alpha = 2 / (period + 1)

    Uses SMA of first `period` valid values as seed (matches TA-Lib behavior).

    Args:
        close: Close prices
        period: Span size
        adjust: Use adjustment factor for initial values

    Returns:
        EMA series
    """
    # Use pandas ewm for vectorized computation
    # span=period gives alpha = 2/(period+1)
    return close.ewm(span=period, adjust=adjust).mean()


def dema(close: pd.Series, period: int) -> pd.Series:
    """
    Double Exponential Moving Average.

    Formula: DEMA = 2 * EMA(close) - EMA(EMA(close))

    Reduces lag compared to single EMA.

    Args:
        close: Close prices
        period: Span size

    Returns:
        DEMA series
    """
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def kama(
    close: pd.Series,
    period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average.

    Uses Efficiency Ratio (ER) to adaptively adjust smoothing:
    - Trending market (ER high): use fast period
    - Choppy market (ER low): use slow period

    ER = |close[t] - close[t-period]| / sum(|close[i] - close[i-1]|, i=t-period+1..t)

    Args:
        close: Close prices
        period: Efficiency ratio period
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        KAMA series
    """
    # Compute efficiency ratio
    direction = (close - close.shift(period)).abs()
    volatility = (close - close.shift(1)).abs().rolling(window=period).sum()
    er = direction / volatility

    # Compute smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)

    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Compute KAMA iteratively. Pre-fill the whole series with NaN so that
    # short series (fewer than ``period`` observations, e.g. newly listed
    # stocks) stay length-aligned with ``close`` instead of raising an
    # out-of-bounds index on ``close.iloc[period]``.
    kama_values = [np.nan] * len(close)
    if len(close) > period:
        kama_values[period] = close.iloc[period]
        for i in range(period + 1, len(close)):
            prev_kama = kama_values[i - 1]
            kama_values[i] = prev_kama + sc.iloc[i] * (close.iloc[i] - prev_kama)

    return pd.Series(kama_values, index=close.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    RSI = 100 - 100 / (1 + RS)
    where RS = avg_gain / avg_loss

    Uses Wilder's smoothing (EMA with alpha=1/period).

    Args:
        close: Close prices
        period: Window size (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - 100 / (1 + rs)

    return rsi_values


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
        close: Close prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(close, fast_period)
    ema_slow = ema(close, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change.

    ROC = (close[t] - close[t-period]) / close[t-period] * 100

    Args:
        close: Close prices
        period: Lookback period

    Returns:
        ROC series (percentage)
    """
    return ((close - close.shift(period)) / close.shift(period)) * 100


def compute_all_momentum(
    close: pd.Series,
    periods: list[int],
) -> dict[str, pd.Series]:
    """
    Compute all momentum indicators for multiple periods.

    Args:
        close: Close prices
        periods: List of periods

    Returns:
        Dict mapping indicator_name to series
    """
    results = {}

    # SMA and EMA for each period
    for period in periods:
        results[f"sma_{period}"] = sma(close, period)
        results[f"ema_{period}"] = ema(close, period)
        results[f"dema_{period}"] = dema(close, period)

    # RSI for each period
    for period in periods:
        results[f"rsi_{period}"] = rsi(close, period)

    # ROC for each period
    for period in periods:
        results[f"roc_{period}"] = roc(close, period)

    # KAMA for each period
    for period in periods:
        results[f"kama_{period}"] = kama(close, period)

    # MACD (single set of parameters)
    macd_line, signal_line, histogram = macd(close)
    results["macd_line"] = macd_line
    results["macd_signal"] = signal_line
    results["macd_histogram"] = histogram

    return results
