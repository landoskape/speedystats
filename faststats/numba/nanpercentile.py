from typing import Tuple
import numba as nb
import numpy as np


def get_nanpercentile(data: np.ndarray, keep_axes: Tuple[int], q) -> np.ndarray:
    if keep_axes == (0,):
        return numba_nanpercentile_keep0(data, q)
    if keep_axes == (1,):
        return numba_nanpercentile_keep1(data, q)
    if keep_axes == (2,):
        return numba_nanpercentile_keep2(data, q)
    if keep_axes == (3,):
        return numba_nanpercentile_keep3(data, q)
    if keep_axes == (4,):
        return numba_nanpercentile_keep4(data, q)
    if keep_axes == (0, 1):
        return numba_nanpercentile_keep01(data, q)
    if keep_axes == (0, 2):
        return numba_nanpercentile_keep02(data, q)
    if keep_axes == (0, 3):
        return numba_nanpercentile_keep03(data, q)
    if keep_axes == (0, 4):
        return numba_nanpercentile_keep04(data, q)
    if keep_axes == (1, 2):
        return numba_nanpercentile_keep12(data, q)
    if keep_axes == (1, 3):
        return numba_nanpercentile_keep13(data, q)
    if keep_axes == (1, 4):
        return numba_nanpercentile_keep14(data, q)
    if keep_axes == (2, 3):
        return numba_nanpercentile_keep23(data, q)
    if keep_axes == (2, 4):
        return numba_nanpercentile_keep24(data, q)
    if keep_axes == (3, 4):
        return numba_nanpercentile_keep34(data, q)
    if keep_axes == (0, 1, 2):
        return numba_nanpercentile_keep012(data, q)
    if keep_axes == (0, 1, 3):
        return numba_nanpercentile_keep013(data, q)
    if keep_axes == (0, 1, 4):
        return numba_nanpercentile_keep014(data, q)
    if keep_axes == (0, 2, 3):
        return numba_nanpercentile_keep023(data, q)
    if keep_axes == (0, 2, 4):
        return numba_nanpercentile_keep024(data, q)
    if keep_axes == (0, 3, 4):
        return numba_nanpercentile_keep034(data, q)
    if keep_axes == (1, 2, 3):
        return numba_nanpercentile_keep123(data, q)
    if keep_axes == (1, 2, 4):
        return numba_nanpercentile_keep124(data, q)
    if keep_axes == (1, 3, 4):
        return numba_nanpercentile_keep134(data, q)
    if keep_axes == (2, 3, 4):
        return numba_nanpercentile_keep234(data, q)
    if keep_axes == (0, 1, 2, 3):
        return numba_nanpercentile_keep0123(data, q)
    if keep_axes == (0, 1, 2, 4):
        return numba_nanpercentile_keep0124(data, q)
    if keep_axes == (0, 1, 3, 4):
        return numba_nanpercentile_keep0134(data, q)
    if keep_axes == (0, 2, 3, 4):
        return numba_nanpercentile_keep0234(data, q)
    if keep_axes == (1, 2, 3, 4):
        return numba_nanpercentile_keep1234(data, q)
    raise ValueError(f"Invalid data shape for nanpercentile, received: {keep_axes}")


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep0(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0,)"""
    output = np.zeros((data.shape[0]))
    for n0 in nb.prange(data.shape[0]):
        output[n0] = np.nanpercentile(data[n0, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep1(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1,)"""
    output = np.zeros((data.shape[1]))
    for n0 in nb.prange(data.shape[1]):
        output[n0] = np.nanpercentile(data[:, n0, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep2(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (2,)"""
    output = np.zeros((data.shape[2]))
    for n0 in nb.prange(data.shape[2]):
        output[n0] = np.nanpercentile(data[:, :, n0, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep3(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (3,)"""
    output = np.zeros((data.shape[3]))
    for n0 in nb.prange(data.shape[3]):
        output[n0] = np.nanpercentile(data[:, :, :, n0, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep4(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (4,)"""
    output = np.zeros((data.shape[4]))
    for n0 in nb.prange(data.shape[4]):
        output[n0] = np.nanpercentile(data[:, :, :, :, n0, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep01(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1)"""
    output = np.zeros((data.shape[0], data.shape[1]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            output[n0, n1] = np.nanpercentile(data[n0, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep02(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 2)"""
    output = np.zeros((data.shape[0], data.shape[2]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            output[n0, n1] = np.nanpercentile(data[n0, :, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep03(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 3)"""
    output = np.zeros((data.shape[0], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[3]):
            output[n0, n1] = np.nanpercentile(data[n0, :, :, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep04(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 4)"""
    output = np.zeros((data.shape[0], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanpercentile(data[n0, :, :, :, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep12(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 2)"""
    output = np.zeros((data.shape[1], data.shape[2]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            output[n0, n1] = np.nanpercentile(data[:, n0, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep13(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 3)"""
    output = np.zeros((data.shape[1], data.shape[3]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[3]):
            output[n0, n1] = np.nanpercentile(data[:, n0, :, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep14(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 4)"""
    output = np.zeros((data.shape[1], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanpercentile(data[:, n0, :, :, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep23(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (2, 3)"""
    output = np.zeros((data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[2]):
        for n1 in nb.prange(data.shape[3]):
            output[n0, n1] = np.nanpercentile(data[:, :, n0, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep24(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (2, 4)"""
    output = np.zeros((data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[2]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanpercentile(data[:, :, n0, :, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep34(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (3, 4)"""
    output = np.zeros((data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[3]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanpercentile(data[:, :, :, n0, n1, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep012(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1, 2)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[2]):
                output[n0, n1, n2] = np.nanpercentile(data[n0, n1, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep013(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1, 3)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[3]):
                output[n0, n1, n2] = np.nanpercentile(data[n0, n1, :, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep014(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1, 4)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanpercentile(data[n0, n1, :, :, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep023(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 2, 3)"""
    output = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                output[n0, n1, n2] = np.nanpercentile(data[n0, :, n1, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep024(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 2, 4)"""
    output = np.zeros((data.shape[0], data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanpercentile(data[n0, :, n1, :, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep034(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 3, 4)"""
    output = np.zeros((data.shape[0], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[3]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanpercentile(data[n0, :, :, n1, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep123(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 2, 3)"""
    output = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                output[n0, n1, n2] = np.nanpercentile(data[:, n0, n1, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep124(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 2, 4)"""
    output = np.zeros((data.shape[1], data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanpercentile(data[:, n0, n1, :, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep134(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 3, 4)"""
    output = np.zeros((data.shape[1], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[3]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanpercentile(data[:, n0, :, n1, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep234(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (2, 3, 4)"""
    output = np.zeros((data.shape[2], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[2]):
        for n1 in nb.prange(data.shape[3]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanpercentile(data[:, :, n0, n1, n2, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep0123(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1, 2, 3)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[2]):
                for n3 in nb.prange(data.shape[3]):
                    output[n0, n1, n2, n3] = np.nanpercentile(data[n0, n1, n2, n3, q])
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep0124(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1, 2, 4)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[2]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanpercentile(
                        data[n0, n1, n2, :, n3, q]
                    )
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep0134(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 1, 3, 4)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[3]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanpercentile(
                        data[n0, n1, :, n2, n3, q]
                    )
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep0234(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (0, 2, 3, 4)"""
    output = np.zeros((data.shape[0], data.shape[2], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanpercentile(
                        data[n0, :, n1, n2, n3, q]
                    )
    return output


@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_nanpercentile_keep1234(data: np.ndarray, q) -> np.ndarray:
    """Numba speedup for nanpercentile reducing all but axes (1, 2, 3, 4)"""
    output = np.zeros((data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanpercentile(
                        data[:, n0, n1, n2, n3, q]
                    )
    return output
