import numpy as np
from faststats import nansum, nanmean, nanstd, nanvar


def test_nansum(sample_with_nan):
    assert np.allclose(nansum(sample_with_nan, axis=0), np.nansum(sample_with_nan, axis=0), equal_nan=True)


def test_nanmean(sample_with_nan):
    assert np.allclose(nanmean(sample_with_nan, axis=0), np.nanmean(sample_with_nan, axis=0), equal_nan=True)


def test_nanstd(sample_with_nan):
    assert np.allclose(nanstd(sample_with_nan, axis=1), np.nanstd(sample_with_nan, axis=1), equal_nan=True)


def test_nanvar(sample_with_nan):
    assert np.allclose(nanvar(sample_with_nan, axis=1), np.nanvar(sample_with_nan, axis=1), equal_nan=True)


def test_all_nan():
    all_nan = np.full((3, 3), np.nan)
    assert np.isnan(nanmean(all_nan)).all()
