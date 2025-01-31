import numpy as np
import pytest


@pytest.fixture
def random_1d():
    np.random.seed(42)
    return np.random.randn(10)


@pytest.fixture
def random_1d_with_nan():
    np.random.seed(42)
    data = np.random.randn(10)
    data[np.random.randint(0, 10)] = np.nan
    return data


@pytest.fixture
def random_2d():
    np.random.seed(42)
    return np.random.randn(10, 10)


@pytest.fixture
def random_2d_with_nan():
    np.random.seed(42)
    data = np.random.randn(10, 10)
    data[np.random.randint(0, 10), np.random.randint(0, 10)] = np.nan
    return data


@pytest.fixture
def random_3d():
    np.random.seed(42)
    return np.random.randn(10, 10, 10)


@pytest.fixture
def random_3d_with_nan():
    np.random.seed(42)
    data = np.random.randn(10, 10, 10)
    data[
        np.random.randint(0, 10),
        np.random.randint(0, 10),
        np.random.randint(0, 10),
    ] = np.nan
    return data
