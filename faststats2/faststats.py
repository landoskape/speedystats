from typing import Union, Iterable, Optional
import numpy as np
from .routing import faststats_route, get_max_dims, get_keep_axes


MAX_DIMS = get_max_dims()


def _call_faststats(
    data: np.ndarray,
    method: str,
    axis: Union[int, Iterable[int]] = -1,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:

    # Identify the shape of the data and the axes to keep
    data_ndims = data.ndim
    data_shape = data.shape
    keep_axes = get_keep_axes(axis, data_ndims)

    # If the number of axes to keep is greater than the max supported, use the numpy method directly
    if any(k >= MAX_DIMS for k in keep_axes):
        np_method = getattr(np, method)
        has_q_param = faststats_route(method)[1]
        if has_q_param:
            return np_method(data, axis, keepdims, q)
        else:
            return np_method(data, axis, keepdims)

    # Reshape the data to be flattened along reducing axes
    last_axis = keep_axes[-1]
    if data_ndims > last_axis + 1:
        new_shape = data_shape[: last_axis + 1] + (-1,)
        data = np.reshape(data, new_shape)

    # Get the numba implementation and check if it has a q parameter
    func, has_q_param = faststats_route(method)

    # Call the numba implementation
    if has_q_param:
        out = func(data, keep_axes, q)
    else:
        out = func(data, keep_axes)

    # Reshape the output to match the original data shape if keepdims is True
    if keepdims:
        out = np.expand_dims(out, axis)

    return out
