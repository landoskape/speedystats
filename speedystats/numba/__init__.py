"""Numba-accelerated statistical functions."""

from .sum import get_sum
from .nansum import get_nansum
from .ptp import get_ptp
from .percentile import get_percentile
from .nanpercentile import get_nanpercentile
from .quantile import get_quantile
from .nanquantile import get_nanquantile
from .median import get_median
from .nanmedian import get_nanmedian
from .average import get_average
from .mean import get_mean
from .nanmean import get_nanmean
from .std import get_std
from .nanstd import get_nanstd
from .var import get_var
from .nanvar import get_nanvar
