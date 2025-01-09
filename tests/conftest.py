import numpy as np
import pytest


@pytest.fixture
def sample_1d():
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_2d():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def sample_with_nan():
    return np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])


@pytest.fixture
def large_random():
    np.random.seed(42)
    return np.random.randn(100, 100)
