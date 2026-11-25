"""Return indicators: simple_return, log_return."""
import pandas as pd
import numpy as np


def compute_simple_return(close: pd.Series, pre_close: pd.Series) -> pd.Series:
    """
    Simple return: (close - pre_close) / pre_close

    Args:
        close: Current day close prices
        pre_close: Previous day close prices

    Returns:
        Simple return series
    """
    return (close - pre_close) / pre_close


def compute_log_return(close: pd.Series, pre_close: pd.Series) -> pd.Series:
    """
    Log return: ln(close / pre_close)

    Properties:
    - Time-additive: sum(log_returns[t:t+n]) == log(close[t+n]/close[t])
    - More normally distributed than simple returns
    - Better for portfolio optimization

    Args:
        close: Current day close prices
        pre_close: Previous day close prices

    Returns:
        Log return series
    """
    return np.log(close / pre_close)


def verify_log_return_additivity(log_returns: pd.Series, close: pd.Series) -> bool:
    """
    Verify log return additivity property.

    For any period [t, t+n]:
    sum(log_returns[t:t+n]) == log(close[t+n] / close[t])

    Args:
        log_returns: Log return series
        close: Close price series

    Returns:
        True if additivity holds within tolerance (1e-10)
    """
    # Skip NaN values
    valid_mask = log_returns.notna() & close.notna()
    if valid_mask.sum() < 10:
        return False

    # Test multiple windows
    for window in [5, 10, 20]:
        rolled_sum = log_returns.rolling(window).sum()
        ratio = close / close.shift(window)
        expected = np.log(ratio)

        # Compare
        diff = (rolled_sum - expected).abs()
        if diff.max() > 1e-10:
            return False

    return True
