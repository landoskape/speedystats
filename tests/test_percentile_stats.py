import numpy as np
from faststats import percentile, nanpercentile, quantile, nanquantile


def test_percentile(sample_2d):
    q = 75
    assert np.allclose(
        percentile(sample_2d, q, axis=0), np.percentile(sample_2d, q, axis=0)
    )


def test_quantile(sample_2d):
    q = 0.75
    assert np.allclose(
        quantile(sample_2d, q, axis=1), np.quantile(sample_2d, q, axis=1)
    )


def test_nanpercentile(sample_with_nan):
    q = 50
    assert np.allclose(
        nanpercentile(sample_with_nan, q, axis=0),
        np.nanpercentile(sample_with_nan, q, axis=0),
        equal_nan=True,
    )


def test_nanquantile(sample_with_nan):
    q = 0.5
    assert np.allclose(
        nanquantile(sample_with_nan, q, axis=1),
        np.nanquantile(sample_with_nan, q, axis=1),
        equal_nan=True,
    )
