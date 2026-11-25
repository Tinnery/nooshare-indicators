"""Cross-section standardization: rank and z-score across all stocks for each date."""
import pandas as pd
import numpy as np
from typing import Optional


def cross_section_rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Compute cross-sectional rank (percentile).

    Rank = rank(value) / count

    Args:
        values: Series of indicator values for a single date
        ascending: If True, higher values get higher ranks

    Returns:
        Rank series (0-1)
    """
    if values.empty:
        return values

    ranked = values.rank(pct=True, ascending=ascending)
    return ranked


def cross_section_zscore(values: pd.Series) -> pd.Series:
    """
    Compute cross-sectional z-score.

    Z = (value - mean) / std

    Uses sample standard deviation (ddof=1) for unbiased estimation.
    Returns NaN when standard deviation is zero or undefined (N < 2).

    Args:
        values: Series of indicator values for a single date

    Returns:
        Z-score series
    """
    if values.empty:
        return values

    # Need at least 2 values for sample std
    if len(values) < 2:
        return pd.Series(np.nan, index=values.index)

    mean = values.mean()
    std = values.std(ddof=1)

    # Zero std or NaN std → return NaN (not 0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=values.index)

    return (values - mean) / std


def compute_cross_section_stats(values: pd.Series) -> dict:
    """
    Compute cross-section statistics.

    Args:
        values: Series of indicator values for a single date

    Returns:
        Dict with count, mean, std, min, q25, median, q75, max
    """
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }

    return {
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(ddof=1),  # Sample std for consistency
        "min": values.min(),
        "q25": values.quantile(0.25),
        "median": values.median(),
        "q75": values.quantile(0.75),
        "max": values.max(),
    }


def add_cross_section_metrics(
    df: pd.DataFrame,
    indicator_cols: list[str],
    date_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    """
    Add cross-section rank and z-score for each indicator column.

    For each date, compute rank and z-score across all stocks.

    Args:
        df: DataFrame with indicator values
        indicator_cols: List of indicator column names
        date_col: Date column name
        code_col: Code column name

    Returns:
        DataFrame with added rank and zscore columns
    """
    df = df.copy()

    for col in indicator_cols:
        rank_col = f"{col}_rank"
        zscore_col = f"{col}_zscore"

        # Initialize columns
        df[rank_col] = np.nan
        df[zscore_col] = np.nan

        # Process each date
        for date in df[date_col].unique():
            mask = df[date_col] == date
            values = df.loc[mask, col].dropna()

            if len(values) > 0:
                # Compute rank
                ranks = cross_section_rank(values)
                df.loc[mask, rank_col] = df.loc[mask, code_col].map(
                    lambda code: ranks.get(code, np.nan)
                    if code in ranks.index
                    else np.nan
                )

                # Compute z-score
                zscores = cross_section_zscore(values)
                df.loc[mask, zscore_col] = df.loc[mask, code_col].map(
                    lambda code: zscores.get(code, np.nan)
                    if code in zscores.index
                    else np.nan
                )

    return df


def compute_cross_section_stats_by_date(
    df: pd.DataFrame,
    indicator_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute cross-section statistics for each date.

    Args:
        df: DataFrame with indicator values
        indicator_col: Indicator column name
        date_col: Date column name

    Returns:
        DataFrame with columns: date, indicator_name, count, mean, std, min, q25, median, q75, max
    """
    results = []

    for date in df[date_col].unique():
        values = df.loc[df[date_col] == date, indicator_col].dropna()
        stats = compute_cross_section_stats(values)
        stats["date"] = date
        stats["indicator_name"] = indicator_col
        results.append(stats)

    return pd.DataFrame(results)
