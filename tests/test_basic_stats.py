import numpy as np
from faststats import sum, mean, std, var, ptp, average


def test_sum(sample_1d, sample_2d):
    assert np.allclose(sum(sample_1d), np.sum(sample_1d))
    assert np.allclose(sum(sample_2d, axis=0), np.sum(sample_2d, axis=0))
    assert np.allclose(sum(sample_2d, axis=1), np.sum(sample_2d, axis=1))


def test_mean(sample_1d, sample_2d):
    assert np.allclose(mean(sample_1d), np.mean(sample_1d))
    assert np.allclose(mean(sample_2d, axis=0), np.mean(sample_2d, axis=0))
    assert np.allclose(mean(sample_2d, axis=1), np.mean(sample_2d, axis=1))


def test_std(sample_1d, sample_2d):
    assert np.allclose(std(sample_1d), np.std(sample_1d))
    assert np.allclose(std(sample_2d, axis=0), np.std(sample_2d, axis=0))
    assert np.allclose(std(sample_2d, axis=1), np.std(sample_2d, axis=1))


def test_var(sample_1d, sample_2d):
    assert np.allclose(var(sample_1d), np.var(sample_1d))
    assert np.allclose(var(sample_2d, axis=0), np.var(sample_2d, axis=0))
    assert np.allclose(var(sample_2d, axis=1), np.var(sample_2d, axis=1))


def test_ptp(sample_1d, sample_2d):
    assert np.allclose(ptp(sample_1d), np.ptp(sample_1d))
    assert np.allclose(ptp(sample_2d, axis=0), np.ptp(sample_2d, axis=0))
    assert np.allclose(ptp(sample_2d, axis=1), np.ptp(sample_2d, axis=1))


def test_average(sample_1d, sample_2d):
    assert np.allclose(average(sample_1d), np.average(sample_1d))
    assert np.allclose(average(sample_2d, axis=0), np.average(sample_2d, axis=0))
    assert np.allclose(average(sample_2d, axis=1), np.average(sample_2d, axis=1))


def test_keepdims(sample_2d):
    result = mean(sample_2d, axis=0, keepdims=True)
    expected = np.mean(sample_2d, axis=0, keepdims=True)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)
