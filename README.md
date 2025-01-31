# FastStats

FastStats is a Python package designed to accelerate NumPy statistical operations using Numba. The package maintains a clean API that mirrors NumPy's interface while providing significant performance improvements through parallel processing.

> "How can I use numba to speed up my median computation?" 
>
> Yep. Just use faststats.

> "Is it possible to use numba to make np.std go faster? 
>
> Yep. Just use faststats.

## Features

- Drop-in replacement for common NumPy statistical functions
- Significant performance improvements through Numba optimization
- Maintains NumPy-like API for easy integration

## Installation
```bash
pip install faststats
```

## Quick Start
```python
import numpy as np
import faststats as fs

# Create some test data
data = np.random.randn(1000, 1000)

# Use just like numpy
mean = fs.mean(data, axis=0)
std = fs.std(data, axis=1)
median = fs.median(data)

# Works with nan values too
data_with_nans = data.copy()
data_with_nans[0, 0] = np.nan
nanmean = fs.nanmean(data_with_nans, axis=0)
```

## Available Functions

- Basic Statistics: `mean`, `median`, `std`, `var`, `sum`
- Range Statistics: `ptp` (peak-to-peak)
- Percentile Functions: `percentile`, `quantile`
- NaN-aware Variants: `nanmean`, `nanmedian`, `nanstd`, `nanvar`, `nansum`
- Additional Functions: `average`, `zscore`

## Performance Note

While FastStats is designed for performance, the actual speedup depends on your specific use case, data size, and hardware. The package is most effective with:
- Large arrays (typically > 100,000 elements)
- Multi-core processors (for parallel execution)
- Certain methods are sped up much more than numpy
- Certain axis / dimension combinations get huge speedups, others are usually comparable to numpy

> **Note:** Benchmarking tools for automatic routing to faststats implementation vs numpy default methods isn't finished --- so you are responsible for determining whether faststats is faster. Here's an example of how to test it quickly: 
>```python
>from time import time
>import numpy as mp
>import faststats as fs
>
># Suppose you want to test if median is faster for arrays of a certain shape and size
>array = np.random.randn(1000, 10000)
>axis = 1
>
># Set repeats to get a better estimate
>num_repeats = 20
>
>t = time()
>for _ in range(num_repeats):
>    _ = np.median(array, axis=axis)
>numpy_time = time() - t
>
>t = time()
>for _ in range(num_repeats):
>    _ = fs.median(array, axis=axis)
>faststats_time = time() - t
>
>print("Speedup: ", numpy_time / faststats_time)
>```

Comprehensive benchmarking tools are under development.

## Development Status

This package is in beta. While the core functionality is stable, we're actively working on:
- Comprehensive benchmarking suite
- Performance optimization guides
- Additional statistical functions
- Advanced documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the LICENSE file for details.