import numpy as np
import time
from typing import Tuple, List, Optional, Union
import faststats
from dataclasses import dataclass
from tabulate import tabulate
from argparse import ArgumentParser


@dataclass
class ArrayConfig:
    shape: Tuple[int, ...]
    name: str


@dataclass
class BenchmarkResult:
    array_name: str
    axis: Union[int, Tuple[int, ...], None]
    numpy_time: float
    faststats_time: float
    speedup: float
    array_size_mb: float


def get_array_configs() -> List[ArrayConfig]:
    """Define different array configurations to test"""
    return [
        # Large arrays
        ArrayConfig((1000, 1000), "large_square"),
        ArrayConfig((1000, 10000), "large_tilted_right"),
        ArrayConfig((10000, 1000), "large_tilted_left"),
        ArrayConfig((10, 100000), "extra_tilted_right"),
        ArrayConfig((100000, 10), "extra_tilted_left"),
        # Extra large arrays
        ArrayConfig((8000, 8000), "huge_square"),
        # Triple arrays
        ArrayConfig((100, 100, 100), "3d_small"),
        ArrayConfig((100, 100, 1000), "3d_small_tilted_right"),
        ArrayConfig((100, 1000, 100), "3d_small_tilted_left"),
        ArrayConfig((100, 1000, 1000), "3d_large_tilted_right"),
        ArrayConfig((1000, 1000, 100), "3d_large_tilted_left"),
    ]


def benchmark_operation(
    data: np.ndarray,
    method: str,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    n_repeats: int = 5,
) -> Tuple[float, float]:
    """Benchmark a single operation comparing NumPy vs FastStats"""

    # Get the corresponding functions
    numpy_func = getattr(np, method)

    faststats_func = getattr(faststats, method)

    # Warm-up run
    _ = numpy_func(data, axis=axis)
    _ = faststats_func(data, axis=axis)

    # Time NumPy
    numpy_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = numpy_func(data, axis=axis)
        numpy_times.append(time.perf_counter() - start)

    # Time FastStats
    faststats_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = faststats_func(data, axis=axis)
        faststats_times.append(time.perf_counter() - start)

    return np.mean(numpy_times), np.mean(faststats_times)


def run_benchmarks(method: str) -> List[BenchmarkResult]:
    """Run comprehensive benchmarks for a given statistical method"""

    configs = get_array_configs()
    axes_to_test = [0, 1, 2, (0, 1), (0, 2), (1, 2)]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"Benchmarking {method} on {config.name} ({i}/{len(configs)})")

        # Create random array
        data = np.random.randn(*config.shape)
        array_size_mb = data.nbytes / (1024 * 1024)

        # Test different axis configurations
        for axis in axes_to_test:
            # Skip invalid axis combinations
            if axis is None:
                continue
            if isinstance(axis, (int, tuple)) and axis is not None:
                if isinstance(axis, int) and axis >= len(config.shape):
                    continue
                if isinstance(axis, tuple) and any(
                    a >= len(config.shape) for a in axis
                ):
                    continue
                if isinstance(axis, tuple) and len(axis) >= len(config.shape):
                    continue
            else:
                continue

            numpy_time, faststats_time = benchmark_operation(
                data,
                method,
                axis,
            )

            results.append(
                BenchmarkResult(
                    array_name=config.name,
                    axis=axis,
                    numpy_time=numpy_time,
                    faststats_time=faststats_time,
                    speedup=numpy_time / faststats_time,
                    array_size_mb=array_size_mb,
                )
            )

    return results


def display_results(results: List[BenchmarkResult]):
    """Display benchmark results in a formatted table"""
    table_data = [
        [
            r.array_name,
            str(r.axis),
            f"{r.array_size_mb:.1f}",
            f"{r.numpy_time*1000:.2f}",
            f"{r.faststats_time*1000:.2f}",
            f"{r.speedup:.2f}x",
        ]
        for r in results
    ]

    headers = [
        "Array",
        "Axis",
        "Size (MB)",
        "NumPy (ms)",
        "FastStats (ms)",
        "Speedup-FS",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("method", default="mean", type=str, help="Method to benchmark")
    args = parser.parse_args()
    results = run_benchmarks(args.method)
    display_results(results)
