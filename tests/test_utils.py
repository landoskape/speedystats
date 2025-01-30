import numpy as np
import pytest
from faststats.utils import (
    _check_iterable,
    _validate_axis,
    _check_axis,
    _get_target_axis,
    _quantile_is_valid,
    _percentile_is_valid,
)


def test_check_iterable():
    assert _check_iterable([1, 2, 3])
    assert _check_iterable(np.array([1, 2, 3]))
    assert not _check_iterable(42)
    assert not _check_iterable(None)


def test_validate_axis():
    assert _validate_axis(0, 3)
    assert _validate_axis(-1, 3)
    assert not _validate_axis(3, 3)
    assert not _validate_axis(-4, 3)


def test_check_axis():
    assert _check_axis(0, 3)
    assert _check_axis(-1, 3)
    assert not _check_axis(3, 3)
    assert not _check_axis(-4, 3)


def test_get_target_axis():
    assert _get_target_axis(0) == -1
    assert np.array_equal(_get_target_axis([0, 1]), np.array([-2, -1]))


def test_quantile_is_valid():
    assert _quantile_is_valid([0.5, 0.75, 0.25])
    assert _quantile_is_valid(0.5)

    with pytest.raises(ValueError):
        _quantile_is_valid([0.5, 0.75, 0.25, 10])
    with pytest.raises(ValueError):
        _quantile_is_valid(10)


def test_percentile_is_valid():
    assert _percentile_is_valid([0.5, 0.75, 0.25])
    assert _percentile_is_valid(50)
    with pytest.raises(ValueError):
        _percentile_is_valid([0.5, 0.75, 0.25, 1000])
    with pytest.raises(ValueError):
        _percentile_is_valid(1000)
