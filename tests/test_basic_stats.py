import numpy as np
import faststats


test_methods = [
    "sum",
    "mean",
    "median",
    "std",
    "var",
    "ptp",
    "average",
]


def test_1d(random_1d):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(faststats_method(random_1d), np_method(random_1d))


def test_2d(random_2d):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(faststats_method(random_2d), np_method(random_2d))
        assert np.allclose(
            faststats_method(random_2d, axis=0), np_method(random_2d, axis=0)
        )
        assert np.allclose(
            faststats_method(random_2d, axis=1), np_method(random_2d, axis=1)
        )


def test_3d(random_3d):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(faststats_method(random_3d), np_method(random_3d))
        assert np.allclose(
            faststats_method(random_3d, axis=0), np_method(random_3d, axis=0)
        )
        assert np.allclose(
            faststats_method(random_3d, axis=1), np_method(random_3d, axis=1)
        )
        assert np.allclose(
            faststats_method(random_3d, axis=2), np_method(random_3d, axis=2)
        )
        assert np.allclose(
            faststats_method(random_3d, axis=(0, 1)), np_method(random_3d, axis=(0, 1))
        )
        assert np.allclose(
            faststats_method(random_3d, axis=(0, 2)), np_method(random_3d, axis=(0, 2))
        )
        assert np.allclose(
            faststats_method(random_3d, axis=(1, 2)), np_method(random_3d, axis=(1, 2))
        )
