from typing import Union, Iterable, Optional
import numpy as np
from .routing import speedystat_route, get_max_dims, get_keep_axes

MAX_DIMS = get_max_dims()


def _call_speedystat(
    data: np.ndarray,
    method: str,
    axis: Optional[Union[int, Iterable[int]]] = None,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:
    # If the axis is None, use the numpy fallback
    if axis is None:
        return _fallback_speedystat(data, method, axis, keepdims, q)

    # Identify the shape of the data and the axes to keep
    data_ndims = data.ndim
    data_shape = data.shape
    keep_axes = get_keep_axes(axis, data_ndims)

    # If no axes are kept, use the numpy fallback
    if not keep_axes:
        return _fallback_speedystat(data, method, axis, keepdims, q)

    # If the number of axes to keep isn't supported, use the numpy fallback
    if any(k >= MAX_DIMS for k in keep_axes):
        return _fallback_speedystat(data, method, axis, keepdims, q)

    # Reshape the data to be flattened along reducing axes
    last_axis = keep_axes[-1]
    if data_ndims > last_axis + 1:
        new_shape = data_shape[: last_axis + 1] + (-1,)
        data = np.reshape(data, new_shape)

    # Get the numba implementation and check if it has a q parameter
    func, has_q_param = speedystat_route(method)

    # Call the numba implementation
    if has_q_param:
        out = func(data, keep_axes, q)
    else:
        out = func(data, keep_axes)

    # Reshape the output to match the original data shape if keepdims is True
    if keepdims:
        out = np.expand_dims(out, axis)

    return out


def _fallback_speedystat(
    data: np.ndarray,
    method: str,
    axis: Optional[Union[int, Iterable[int]]] = None,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:
    np_method = getattr(np, method)
    if q is not None:
        return np_method(data, axis=axis, keepdims=keepdims, q=q)
    else:
        return np_method(data, axis=axis, keepdims=keepdims)


def sum(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "sum", axis, keepdims)


def nansum(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "nansum", axis, keepdims)


def ptp(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "ptp", axis, keepdims)


def percentile(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:
    return _call_speedystat(data, "percentile", axis, keepdims, q)


def nanpercentile(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:
    return _call_speedystat(data, "nanpercentile", axis, keepdims, q)


def quantile(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:
    return _call_speedystat(data, "quantile", axis, keepdims, q)


def nanquantile(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:
    return _call_speedystat(data, "nanquantile", axis, keepdims, q)


def median(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "median", axis, keepdims)


def nanmedian(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "nanmedian", axis, keepdims)


def average(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "average", axis, keepdims)


def mean(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "mean", axis, keepdims)


def nanmean(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "nanmean", axis, keepdims)


def std(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "std", axis, keepdims)


def nanstd(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "nanstd", axis, keepdims)


def var(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "var", axis, keepdims)


def nanvar(
    data: np.ndarray,
    axis: Union[int, Iterable[int]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return _call_speedystat(data, "nanvar", axis, keepdims)
