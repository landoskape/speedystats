from typing import Callable, Union, Iterable, Tuple
from . import numba


def faststats_route(np_method: str) -> Callable:
    """Route numpy method names to their fast implementations.

    Args:
        np_method: Name of the numpy method to route

    Returns:
        Callable: The corresponding fast implementation
    """
    method_map = {
        "sum": numba.sum.get_sum,
        "nansum": numba.nansum.get_nansum,
        "ptp": numba.ptp.get_ptp,
        "percentile": numba.percentile.get_percentile,
        "nanpercentile": numba.nanpercentile.get_nanpercentile,
        "quantile": numba.quantile.get_quantile,
        "nanquantile": numba.nanquantile.get_nanquantile,
        "median": numba.median.get_median,
        "nanmedian": numba.nanmedian.get_nanmedian,
        "average": numba.average.get_average,
        "mean": numba.mean.get_mean,
        "nanmean": numba.nanmean.get_nanmean,
        "std": numba.std.get_std,
        "nanstd": numba.nanstd.get_nanstd,
        "var": numba.var.get_var,
        "nanvar": numba.nanvar.get_nanvar,
    }
    has_q_param = {
        "sum": False,
        "nansum": False,
        "ptp": False,
        "percentile": True,
        "nanpercentile": True,
        "quantile": True,
        "nanquantile": True,
        "median": False,
        "nanmedian": False,
        "average": False,
        "mean": False,
        "nanmean": False,
        "std": False,
        "nanstd": False,
        "var": False,
        "nanvar": False,
    }

    if np_method not in method_map:
        raise ValueError(f"No fast implementation available for {np_method}")
    if np_method not in has_q_param:
        raise ValueError(f"No q param config available for {np_method}")
    return method_map[np_method], has_q_param[np_method]


def get_max_dims() -> int:
    """Get the maximum number of dimensions supported by the fast implementations.

    Returns:
        int: Maximum number of dimensions supported
    """
    return 5


def get_keep_axes(axis: Union[int, Iterable[int]], ndim: int) -> Tuple[int]:
    keep_axes = list(range(ndim))
    if isinstance(axis, int):
        axis = [axis]
    for a in axis:
        keep_axes.remove(a)

    keep_axes = tuple(keep_axes)
    return keep_axes
