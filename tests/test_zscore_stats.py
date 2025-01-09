import numpy as np
from faststats import zscore, nan_zscore


def test_zscore(sample_2d):
    mean = np.mean(sample_2d, axis=1, keepdims=True)
    std = np.std(sample_2d, axis=1, keepdims=True)
    expected = (sample_2d - mean) / std
    print(sample_2d)
    print(expected)
    print(sample_2d)
    print(zscore(sample_2d, axis=1))
    assert np.allclose(zscore(sample_2d, axis=1), expected)


def test_nan_zscore(sample_with_nan):
    result = nan_zscore(sample_with_nan, axis=0)
    mean = np.nanmean(sample_with_nan, axis=0, keepdims=True)
    std = np.nanstd(sample_with_nan, axis=0, keepdims=True)
    expected = (sample_with_nan - mean) / std
    assert np.allclose(result, expected, equal_nan=True)
