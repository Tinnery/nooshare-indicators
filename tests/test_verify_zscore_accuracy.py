"""Z-score 精度校验器的独立参考实现用的测试数据集。"""

import numpy as np
import pandas as pd

from scripts.verify_zscore_accuracy import (
    compute_volume_zscore_independent,
    compute_zscore_independent,
)


def test_volume_reference_excludes_current_from_baseline():
    volume = pd.Series([10.0, 20.0, 30.0, 40.0], name="volume")

    result = compute_volume_zscore_independent(volume, window=3)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(2.0)


def test_volume_reference_uses_prior_valid_positive_finite_observations():
    volume = pd.Series(
        [10.0, 0.0, 20.0, np.nan, 30.0, np.inf, 40.0],
        index=[1, 1, 2, 2, 3, 3, 4],
        name="volume",
    )

    result = compute_volume_zscore_independent(volume, window=3)

    assert result.index.equals(volume.index)
    assert result.name == "volume"
    assert result.iloc[1] != result.iloc[1]  # zero current is invalid
    assert result.iloc[3] != result.iloc[3]  # NaN current is invalid
    assert result.iloc[5] != result.iloc[5]  # infinite current is invalid
    assert result.iloc[6] == pytest.approx(2.0)


def test_volume_reference_requires_prior_history_and_nonzero_sample_std():
    insufficient = pd.Series([10.0, 20.0, 30.0])
    zero_std = pd.Series([10.0, 10.0, 10.0, 20.0])

    assert compute_volume_zscore_independent(insufficient, 3).isna().all()
    assert pd.isna(compute_volume_zscore_independent(zero_std, 3).iloc[-1])


def test_close_reference_keeps_its_existing_current_inclusive_window():
    close = pd.Series([10.0, 20.0, 30.0, 40.0])

    result = compute_zscore_independent(close, window=3)

    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(1.0)


import pytest
