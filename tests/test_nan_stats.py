import numpy as np
import faststats


test_methods = [
    "nansum",
    "nanmean",
    "nanmedian",
    "nanstd",
    "nanvar",
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
            faststats_method(random_2d, axis=0),
            np_method(random_2d, axis=0),
        )
        assert np.allclose(
            faststats_method(random_2d, axis=1),
            np_method(random_2d, axis=1),
        )


def test_3d(random_3d):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(faststats_method(random_3d), np_method(random_3d))
        assert np.allclose(
            faststats_method(random_3d, axis=0),
            np_method(random_3d, axis=0),
        )
        assert np.allclose(
            faststats_method(random_3d, axis=1),
            np_method(random_3d, axis=1),
        )
        assert np.allclose(
            faststats_method(random_3d, axis=2),
            np_method(random_3d, axis=2),
        )
        assert np.allclose(
            faststats_method(random_3d, axis=(0, 1)),
            np_method(random_3d, axis=(0, 1)),
        )
        assert np.allclose(
            faststats_method(random_3d, axis=(0, 2)),
            np_method(random_3d, axis=(0, 2)),
        )
        assert np.allclose(
            faststats_method(random_3d, axis=(1, 2)),
            np_method(random_3d, axis=(1, 2)),
        )


def test_1d_nan(random_1d_with_nan):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(
            faststats_method(random_1d_with_nan), np_method(random_1d_with_nan)
        )


def test_2d_nan(random_2d_with_nan):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(
            faststats_method(random_2d_with_nan), np_method(random_2d_with_nan)
        )
        assert np.allclose(
            faststats_method(random_2d_with_nan, axis=0),
            np_method(random_2d_with_nan, axis=0),
        )
        assert np.allclose(
            faststats_method(random_2d_with_nan, axis=1),
            np_method(random_2d_with_nan, axis=1),
        )


def test_3d_nan(random_3d_with_nan):
    for method in test_methods:
        np_method = getattr(np, method)
        faststats_method = getattr(faststats, method)
        assert np.allclose(
            faststats_method(random_3d_with_nan), np_method(random_3d_with_nan)
        )
        assert np.allclose(
            faststats_method(random_3d_with_nan, axis=0),
            np_method(random_3d_with_nan, axis=0),
        )
        assert np.allclose(
            faststats_method(random_3d_with_nan, axis=1),
            np_method(random_3d_with_nan, axis=1),
        )
        assert np.allclose(
            faststats_method(random_3d_with_nan, axis=2),
            np_method(random_3d_with_nan, axis=2),
        )
        assert np.allclose(
            faststats_method(random_3d_with_nan, axis=(0, 1)),
            np_method(random_3d_with_nan, axis=(0, 1)),
        )
        assert np.allclose(
            faststats_method(random_3d_with_nan, axis=(0, 2)),
            np_method(random_3d_with_nan, axis=(0, 2)),
        )
        assert np.allclose(
            faststats_method(random_3d_with_nan, axis=(1, 2)),
            np_method(random_3d_with_nan, axis=(1, 2)),
        )
