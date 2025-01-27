import numpy as np

# from .routing import faststats_route, get_max_dims

# from .numba import get_keep_axes


def faststat(data, method, axis=-1, keepdims=False, q=None):
    data_ndims = data.ndim
    data_shape = data.shape
    keep_axes = get_keep_axes(axis, data_ndims)
    last_axis = keep_axes[-1]
    if data_ndims > last_axis + 1:
        new_shape = data_shape[:last_axis] + (-1,)
        data = np.reshape(data, new_shape)
    return
