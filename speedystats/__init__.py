"""Numba-accelerated statistical functions."""

__version__ = "0.0.2"


from .speedystats import sum
from .speedystats import nansum
from .speedystats import ptp
from .speedystats import percentile
from .speedystats import nanpercentile
from .speedystats import quantile
from .speedystats import nanquantile
from .speedystats import median
from .speedystats import nanmedian
from .speedystats import average
from .speedystats import mean
from .speedystats import nanmean
from .speedystats import std
from .speedystats import nanstd
from .speedystats import var
from .speedystats import nanvar
