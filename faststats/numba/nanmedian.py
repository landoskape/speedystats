from typing import Tuple
import numba as nb
import numpy as np


def get_nanmedian(data: np.ndarray, keep_axes: Tuple[int]) -> np.ndarray:
    if keep_axes == (0,):
        return numba_nanmedian_keep0(data)
    if keep_axes == (1,):
        return numba_nanmedian_keep1(data)
    if keep_axes == (2,):
        return numba_nanmedian_keep2(data)
    if keep_axes == (3,):
        return numba_nanmedian_keep3(data)
    if keep_axes == (4,):
        return numba_nanmedian_keep4(data)
    if keep_axes == (0, 1):
        return numba_nanmedian_keep01(data)
    if keep_axes == (0, 2):
        return numba_nanmedian_keep02(data)
    if keep_axes == (0, 3):
        return numba_nanmedian_keep03(data)
    if keep_axes == (0, 4):
        return numba_nanmedian_keep04(data)
    if keep_axes == (1, 2):
        return numba_nanmedian_keep12(data)
    if keep_axes == (1, 3):
        return numba_nanmedian_keep13(data)
    if keep_axes == (1, 4):
        return numba_nanmedian_keep14(data)
    if keep_axes == (2, 3):
        return numba_nanmedian_keep23(data)
    if keep_axes == (2, 4):
        return numba_nanmedian_keep24(data)
    if keep_axes == (3, 4):
        return numba_nanmedian_keep34(data)
    if keep_axes == (0, 1, 2):
        return numba_nanmedian_keep012(data)
    if keep_axes == (0, 1, 3):
        return numba_nanmedian_keep013(data)
    if keep_axes == (0, 1, 4):
        return numba_nanmedian_keep014(data)
    if keep_axes == (0, 2, 3):
        return numba_nanmedian_keep023(data)
    if keep_axes == (0, 2, 4):
        return numba_nanmedian_keep024(data)
    if keep_axes == (0, 3, 4):
        return numba_nanmedian_keep034(data)
    if keep_axes == (1, 2, 3):
        return numba_nanmedian_keep123(data)
    if keep_axes == (1, 2, 4):
        return numba_nanmedian_keep124(data)
    if keep_axes == (1, 3, 4):
        return numba_nanmedian_keep134(data)
    if keep_axes == (2, 3, 4):
        return numba_nanmedian_keep234(data)
    if keep_axes == (0, 1, 2, 3):
        return numba_nanmedian_keep0123(data)
    if keep_axes == (0, 1, 2, 4):
        return numba_nanmedian_keep0124(data)
    if keep_axes == (0, 1, 3, 4):
        return numba_nanmedian_keep0134(data)
    if keep_axes == (0, 2, 3, 4):
        return numba_nanmedian_keep0234(data)
    if keep_axes == (1, 2, 3, 4):
        return numba_nanmedian_keep1234(data)
    raise ValueError(f"Invalid data shape for nanmedian, received: {keep_axes}")


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep0(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0,)"""
    output = np.zeros((data.shape[0]))
    for n0 in nb.prange(data.shape[0]):
        output[n0] = np.nanmedian(data[n0])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep1(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1,)"""
    output = np.zeros((data.shape[1]))
    for n0 in nb.prange(data.shape[1]):
        output[n0] = np.nanmedian(data[:, n0])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep2(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (2,)"""
    output = np.zeros((data.shape[2]))
    for n0 in nb.prange(data.shape[2]):
        output[n0] = np.nanmedian(data[:, :, n0])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep3(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (3,)"""
    output = np.zeros((data.shape[3]))
    for n0 in nb.prange(data.shape[3]):
        output[n0] = np.nanmedian(data[:, :, :, n0])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep4(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (4,)"""
    output = np.zeros((data.shape[4]))
    for n0 in nb.prange(data.shape[4]):
        output[n0] = np.nanmedian(data[:, :, :, :, n0])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep01(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1)"""
    output = np.zeros((data.shape[0], data.shape[1]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            output[n0, n1] = np.nanmedian(data[n0, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep02(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 2)"""
    output = np.zeros((data.shape[0], data.shape[2]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            output[n0, n1] = np.nanmedian(data[n0, :, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep03(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 3)"""
    output = np.zeros((data.shape[0], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[3]):
            output[n0, n1] = np.nanmedian(data[n0, :, :, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep04(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 4)"""
    output = np.zeros((data.shape[0], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanmedian(data[n0, :, :, :, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep12(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 2)"""
    output = np.zeros((data.shape[1], data.shape[2]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            output[n0, n1] = np.nanmedian(data[:, n0, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep13(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 3)"""
    output = np.zeros((data.shape[1], data.shape[3]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[3]):
            output[n0, n1] = np.nanmedian(data[:, n0, :, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep14(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 4)"""
    output = np.zeros((data.shape[1], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanmedian(data[:, n0, :, :, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep23(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (2, 3)"""
    output = np.zeros((data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[2]):
        for n1 in nb.prange(data.shape[3]):
            output[n0, n1] = np.nanmedian(data[:, :, n0, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep24(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (2, 4)"""
    output = np.zeros((data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[2]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanmedian(data[:, :, n0, :, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep34(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (3, 4)"""
    output = np.zeros((data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[3]):
        for n1 in nb.prange(data.shape[4]):
            output[n0, n1] = np.nanmedian(data[:, :, :, n0, n1])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep012(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1, 2)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[2]):
                output[n0, n1, n2] = np.nanmedian(data[n0, n1, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep013(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1, 3)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[3]):
                output[n0, n1, n2] = np.nanmedian(data[n0, n1, :, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep014(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1, 4)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanmedian(data[n0, n1, :, :, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep023(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 2, 3)"""
    output = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                output[n0, n1, n2] = np.nanmedian(data[n0, :, n1, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep024(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 2, 4)"""
    output = np.zeros((data.shape[0], data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanmedian(data[n0, :, n1, :, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep034(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 3, 4)"""
    output = np.zeros((data.shape[0], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[3]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanmedian(data[n0, :, :, n1, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep123(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 2, 3)"""
    output = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                output[n0, n1, n2] = np.nanmedian(data[:, n0, n1, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep124(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 2, 4)"""
    output = np.zeros((data.shape[1], data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanmedian(data[:, n0, n1, :, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep134(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 3, 4)"""
    output = np.zeros((data.shape[1], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[3]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanmedian(data[:, n0, :, n1, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep234(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (2, 3, 4)"""
    output = np.zeros((data.shape[2], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[2]):
        for n1 in nb.prange(data.shape[3]):
            for n2 in nb.prange(data.shape[4]):
                output[n0, n1, n2] = np.nanmedian(data[:, :, n0, n1, n2])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep0123(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1, 2, 3)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[2]):
                for n3 in nb.prange(data.shape[3]):
                    output[n0, n1, n2, n3] = np.nanmedian(data[n0, n1, n2, n3])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep0124(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1, 2, 4)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[2]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanmedian(data[n0, n1, n2, :, n3])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep0134(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 1, 3, 4)"""
    output = np.zeros((data.shape[0], data.shape[1], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            for n2 in nb.prange(data.shape[3]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanmedian(data[n0, n1, :, n2, n3])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep0234(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (0, 2, 3, 4)"""
    output = np.zeros((data.shape[0], data.shape[2], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanmedian(data[n0, :, n1, n2, n3])
    return output


@nb.njit(parallel=True, fastmath=False, cache=True)
def numba_nanmedian_keep1234(data: np.ndarray) -> np.ndarray:
    """Numba speedup for nanmedian reducing all but axes (1, 2, 3, 4)"""
    output = np.zeros((data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
    for n0 in nb.prange(data.shape[1]):
        for n1 in nb.prange(data.shape[2]):
            for n2 in nb.prange(data.shape[3]):
                for n3 in nb.prange(data.shape[4]):
                    output[n0, n1, n2, n3] = np.nanmedian(data[:, n0, n1, n2, n3])
    return output
