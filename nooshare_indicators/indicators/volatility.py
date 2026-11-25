"""Volatility indicators: CCC, Parkinson, Garman-Klass, Rogers-Satchell."""
import pandas as pd
import numpy as np


def volatility_close_to_close(log_returns: pd.Series, period: int) -> pd.Series:
    """
    Close-to-Close volatility (CCC).

    Formula: std(log_returns, window=period) * sqrt(252)

    This is the classical volatility estimator using only close prices.
    Annualized by multiplying with sqrt(252) trading days.

    Efficiency: 1.0x (baseline)

    Args:
        log_returns: Log return series
        period: Rolling window size

    Returns:
        Annualized volatility series
    """
    return log_returns.rolling(window=period).std() * np.sqrt(252)


def volatility_parkinson(
    high: pd.Series,
    low: pd.Series,
    period: int,
) -> pd.Series:
    """
    Parkinson volatility estimator.

    Uses high-low range:
    sigma^2 = (1 / (4 * ln(2))) * mean((ln(high/low))^2)

    Efficiency: 5.1x vs CCC (no drift assumption)

    Args:
        high: High prices
        low: Low prices
        period: Rolling window size

    Returns:
        Annualized volatility series
    """
    log_hl = np.log(high / low)
    sigma_sq = (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2).rolling(window=period).mean()
    return np.sqrt(sigma_sq * 252)


def volatility_garman_klass(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    Uses OHLC prices:
    sigma^2 = mean(
        0.5 * (ln(high/low))^2
        - (2*ln(2) - 1) * (ln(close/open))^2
    )

    Efficiency: 7.4x vs CCC (most efficient)

    Args:
        open: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        period: Rolling window size

    Returns:
        Annualized volatility series
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open)

    sigma_sq = (
        0.5 * (log_hl ** 2)
        - (2 * np.log(2) - 1) * (log_co ** 2)
    ).rolling(window=period).mean()

    return np.sqrt(sigma_sq * 252)


def volatility_rogers_satchell(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Rogers-Satchell volatility estimator.

    Accounts for drift (trend):
    sigma^2 = mean(
        ln(high/close) * ln(high/open)
        + ln(low/close) * ln(low/open)
    )

    Efficiency: 5.1x vs CCC (robust with drift)

    Args:
        open: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        period: Rolling window size

    Returns:
        Annualized volatility series
    """
    log_hc = np.log(high / close)
    log_ho = np.log(high / open)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open)

    sigma_sq = (log_hc * log_ho + log_lc * log_lo).rolling(window=period).mean()

    return np.sqrt(sigma_sq * 252)


def compute_all_volatilities(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    log_returns: pd.Series,
    periods: list[int],
    dates: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute all 4 volatility estimators for multiple periods.

    Args:
        open: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        log_returns: Log returns
        periods: List of rolling window sizes
        dates: Date series (optional, for tracking)

    Returns:
        DataFrame with columns: code, date, period, vol_ccc, vol_parkinson, vol_gk, vol_rs
        Each row corresponds to one (period, date) combination
    """
    rows = []
    
    for period in periods:
        vol_ccc = volatility_close_to_close(log_returns, period)
        vol_parkinson = volatility_parkinson(high, low, period)
        vol_gk = volatility_garman_klass(open, high, low, close, period)
        vol_rs = volatility_rogers_satchell(open, high, low, close, period)
        
        # 为每个日期创建一行
        for i in range(len(log_returns)):
            row = {
                'period': period,
                'vol_ccc': vol_ccc.iloc[i],
                'vol_parkinson': vol_parkinson.iloc[i],
                'vol_gk': vol_gk.iloc[i],
                'vol_rs': vol_rs.iloc[i],
            }
            if dates is not None:
                row['date'] = dates.iloc[i]
            rows.append(row)
    
    return pd.DataFrame(rows)
