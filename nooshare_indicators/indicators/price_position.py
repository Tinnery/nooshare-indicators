"""Price position indicators: Bollinger, ATR, N-day high/low, MA deviation."""
import pandas as pd
import numpy as np
from .momentum import sma


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Middle = SMA(close, period)
    Upper = Middle + num_std * std(close, period)
    Lower = Middle - num_std * std(close, period)

    Args:
        close: Close prices
        period: Window size (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bollinger_width(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Bollinger Band Width.

    Width = (upper - lower) / middle

    Measures volatility expansion/contraction.

    Args:
        close: Close prices
        period: Window size
        num_std: Number of standard deviations

    Returns:
        Band width series
    """
    upper, middle, lower = bollinger_bands(close, period, num_std)
    return (upper - lower) / middle


def bollinger_percent(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Bollinger Band Percent.

    %B = (close - lower) / (upper - lower)

    Measures where price is within the bands (0-1 range).

    Args:
        close: Close prices
        period: Window size
        num_std: Number of standard deviations

    Returns:
        %B series
    """
    upper, middle, lower = bollinger_bands(close, period, num_std)
    return (close - lower) / (upper - lower)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range.

    True Range = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )
    ATR = Wilder's smoothing of True Range

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Window size (default 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (EMA with alpha=1/period)
    atr_values = true_range.ewm(alpha=1.0 / period, adjust=False).mean()

    return atr_values


def n_day_high(high: pd.Series, period: int) -> pd.Series:
    """
    N-day high (rolling maximum high).

    Args:
        high: High prices
        period: Window size

    Returns:
        N-day high series
    """
    return high.rolling(window=period).max()


def n_day_low(low: pd.Series, period: int) -> pd.Series:
    """
    N-day low (rolling minimum low).

    Args:
        low: Low prices
        period: Window size

    Returns:
        N-day low series
    """
    return low.rolling(window=period).min()


def ma_deviation(close: pd.Series, period: int) -> pd.Series:
    """
    Deviation from moving average.

    Deviation = (close - SMA) / SMA * 100

    Measures how far price is from the moving average (percentage).

    Args:
        close: Close prices
        period: SMA period

    Returns:
        Deviation series (percentage)
    """
    ma = sma(close, period)
    return ((close - ma) / ma) * 100


def price_position_in_range(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int,
) -> pd.Series:
    """
    Price position within N-day range.

    Position = (close - n_day_low) / (n_day_high - n_day_low)

    Measures where current price is within the N-day range (0-1).

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        period: Window size

    Returns:
        Position series (0-1)
    """
    n_high = n_day_high(high, period)
    n_low = n_day_low(low, period)
    return (close - n_low) / (n_high - n_low)


def compute_all_price_position(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    periods: list[int],
    boll_period: int = 20,
    boll_std: float = 2.0,
    atr_period: int = 14,
) -> dict[str, pd.Series]:
    """
    Compute all price position indicators.

    Args:
        open: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        periods: List of periods for N-day high/low and MA deviation
        boll_period: Bollinger Bands period
        boll_std: Bollinger Bands standard deviations
        atr_period: ATR period

    Returns:
        Dict mapping indicator_name to series
    """
    results = {}

    # Bollinger Bands
    upper, middle, lower = bollinger_bands(close, boll_period, boll_std)
    results[f"boll_upper_{boll_period}"] = upper
    results[f"boll_middle_{boll_period}"] = middle
    results[f"boll_lower_{boll_period}"] = lower
    results[f"boll_width_{boll_period}"] = bollinger_width(close, boll_period, boll_std)
    results[f"boll_percent_{boll_period}"] = bollinger_percent(close, boll_period, boll_std)

    # ATR
    results[f"atr_{atr_period}"] = atr(high, low, close, atr_period)

    # N-day high/low and position for each period
    for period in periods:
        results[f"n_day_high_{period}"] = n_day_high(high, period)
        results[f"n_day_low_{period}"] = n_day_low(low, period)
        results[f"price_position_{period}"] = price_position_in_range(
            close, high, low, period
        )

    # MA deviation for each period
    for period in periods:
        results[f"ma_deviation_{period}"] = ma_deviation(close, period)

    return results


def close_hfq_rolling_zscore(close_hfq: pd.Series, period: int, is_synthetic: pd.Series = None) -> pd.Series:
    """
    后复权收盘价的 Rolling Z-score（C2 语义：最近 N 个有效观察）。

    只使用有效观察计算最近 period 个有效观察（含当日 T）的 Z-score。
    原始索引中无有效观察的日期返回 NaN。

    有效观察 = 有限数值且严格大于 0。NaN、正负 Inf、零价和负价
    永远不进入有效观察窗口，也不会被 forward-fill。

    Z = (close_hfq - mean(valid_close, period)) / std(valid_close, period)

    Args:
        close_hfq: 后复权收盘价序列（可含 NaN / Inf / 非数值）
        period: 滚动窗口大小（有效观察数量）
        is_synthetic: 布尔序列，标记合成日历日（可选）

    Returns:
        Z-score 序列（原始索引，无有效观察日为 NaN）

    Raises:
        ValueError: 索引存在重复标签（重复观察的滚动窗口语义未定义，
            必须显式失败而不是静默产出错误值）。
    """
    if close_hfq.index.has_duplicates:
        raise ValueError(
            "close_hfq_rolling_zscore requires a unique index; duplicate "
            "observations have undefined rolling-window semantics"
        )

    # 保存原始索引
    original_index = close_hfq.index

    # 先做数值转换：非数值条目 coerce 成 NaN 而不是抛异常
    numeric_close = pd.to_numeric(close_hfq, errors="coerce")

    # C2 有效观察：有限且 > 0。NaN / 正负 Inf / 零价 / 负价全部排除。
    valid_mask = np.isfinite(numeric_close) & (numeric_close > 0)
    
    # 如果提供 is_synthetic，合成日历日也排除
    if is_synthetic is not None:
        valid_mask = valid_mask & ~is_synthetic
    
    valid_close = numeric_close.where(valid_mask).dropna()

    # 在有效观察序列上计算 rolling zscore（合成日历日不可能进入窗口：
    # 它们在上方被过滤，而不是被 forward-fill）
    rolling_mean = valid_close.rolling(window=period, min_periods=period).mean()
    rolling_std = valid_close.rolling(window=period, min_periods=period).std(ddof=1)

    zscore = (valid_close - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    # reindex 回原始索引（无有效观察日自动为 NaN）
    result = zscore.reindex(original_index)

    return result