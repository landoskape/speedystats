import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from itertools import combinations, product
import json
import time
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging
from numpyencoder import NumpyEncoder
import faststats

# Setup logging
logging.basicConfig(level=logging.INFO)


@dataclass
class BenchmarkResult:
    shape: Tuple[int, ...]
    axes: Tuple[int, ...]
    total_elements: int
    elements_reduced: int
    elements_kept: int
    reduction_ratio: float
    numpy_time: float
    faststats_time: float
    speedup: float
    method: str


class SmartBenchmarker:
    """
    Intelligent benchmarking system that:
    1. Systematically samples array shapes and reduction patterns
    2. Uses parallel processing for efficiency
    3. Builds a predictive model for speedup
    4. Saves results for future use
    """

    def __init__(
        self,
        max_dims: int = 3,
        max_gb: int = 16,  # GB
        shape_log_base: int = 10,
        shape_power_range: Tuple[int, int] = (1, 4),
        shape_power_step: int = 1,
        n_repeats: int = 5,
        methods: Optional[List[str]] = None,
        results_dir: str = "benchmark_results",
    ):
        self.max_dims = max_dims
        self.max_gb = max_gb
        self.shape_log_base = shape_log_base
        self.shape_power_range = shape_power_range
        self.shape_power_step = shape_power_step
        self.n_repeats = n_repeats
        self.methods = methods or ["mean", "median", "std"]
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.bytes_per_element = 8
        self.max_bytes = self.max_gb * 1024**3

        logging.basicConfig(
            filename=self.results_dir / "benchmark.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            # handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)

    def generate_shapes(self) -> List[Tuple[int, ...]]:
        """
        Generate array shapes using logarithmic sampling to ensure good coverage
        across orders of magnitude.
        """
        shapes = []
        array_nbytes = []
        dim_shapes = self.shape_log_base ** np.arange(
            self.shape_power_range[0],
            self.shape_power_range[1] + self.shape_power_step,
            self.shape_power_step,
        )
        dim_shapes = [int(x) for x in dim_shapes]

        # Generate different dimensionalities
        for ndim in range(2, self.max_dims + 1):
            options = [dim_shapes] * ndim
            for shape in product(*options):
                if np.prod(shape) * self.bytes_per_element <= self.max_bytes:
                    shapes.append(shape)
                    array_nbytes.append(np.prod(shape) * self.bytes_per_element)

        # Sort first by dimension count, then by array size
        shapes = sorted(shapes, key=lambda x: (len(x), array_nbytes[shapes.index(x)]))
        return shapes

    def generate_axes_combinations(self, ndim: int) -> List[Tuple[int, ...]]:
        """Generate all valid reduction axis combinations for given dimensions."""
        all_axes = []
        for r in range(1, ndim):  # r is the number of axes to reduce
            all_axes.extend(combinations(range(ndim), r))
        return all_axes

    def benchmark_single_case(
        self,
        data: np.ndarray,
        shape: Tuple[int, ...],
        axes: Tuple[int, ...],
        method: str,
    ) -> BenchmarkResult:
        """Benchmark a single shape/axes combination."""
        try:
            # Calculate metadata
            total_elements = np.prod(shape)
            elements_kept = np.prod(
                [shape[i] for i in range(len(shape)) if i not in axes]
            )
            elements_reduced = (
                total_elements / elements_kept if elements_kept > 0 else total_elements
            )
            reduction_ratio = elements_reduced / total_elements

            # Run benchmarks
            numpy_times = []
            faststats_times = []

            numpy_func = getattr(np, method)
            faststats_func = getattr(faststats, method)

            for _ in range(self.n_repeats):
                # Time NumPy
                start = time.perf_counter()
                _ = numpy_func(data, axis=axes)
                numpy_times.append(time.perf_counter() - start)

                # Time FastStats
                start = time.perf_counter()
                _ = faststats_func(data, axis=axes)
                faststats_times.append(time.perf_counter() - start)

            return BenchmarkResult(
                shape=shape,
                axes=axes,
                total_elements=total_elements,
                elements_reduced=elements_reduced,
                elements_kept=elements_kept,
                reduction_ratio=reduction_ratio,
                numpy_time=np.mean(numpy_times),
                faststats_time=np.mean(faststats_times),
                speedup=np.mean(numpy_times) / np.mean(faststats_times),
                method=method,
                array_nbytes=np.prod(shape) * self.bytes_per_element,
            )

        except Exception as e:
            self.logger.error(
                f"Error benchmarking shape={shape}, axes={axes}: {str(e)}"
            )
            return None

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks sequentially to ensure accurate Numba timing."""
        shapes = self.generate_shape_samples()
        axes_combinations = {
            shape: self.generate_axes_combinations(len(shape)) for shape in shapes
        }
        num_tests = sum(len(axes) for axes in axes_combinations.values())
        results = []

        # Run benchmarks sequentially
        total = num_tests * len(self.methods)
        current = 0

        for shape in shapes:
            # Setup some warmup data for this number of dimensions with a very small size
            ndim = len(shape)
            warmup_data = np.random.randn(*([5] * ndim))

            axes_combinations = self.generate_axes_combinations(len(shape))
            for axes in axes_combinations:
                data = np.random.randn(*shape)

                # Warmup faststats function for this dimension / axes combination
                for method in self.methods:
                    faststats_func = getattr(faststats, method)
                    _ = faststats_func(warmup_data, axis=axes)

                for method in self.methods:
                    current += 1
                    self.logger.info(
                        f"Running benchmark {current}/{total}: {method} on shape {shape} axes {axes}"
                    )

                    result = self.benchmark_single_case(data, shape, axes, method)
                    if result is not None:
                        results.append(result)

        return results

    def build_predictive_model(self, results: List[BenchmarkResult]):
        """Build a Random Forest model to predict speedup."""
        # Convert results to DataFrame
        data = []
        for r in results:
            data.append(
                {
                    "ndim": len(r.shape),
                    "total_elements": r.total_elements,
                    "elements_reduced": r.elements_reduced,
                    "elements_kept": r.elements_kept,
                    "reduction_ratio": r.reduction_ratio,
                    "method": r.method,
                    "speedup": r.speedup,
                    "array_nbytes": r.array_nbytes,
                }
            )

        df = pd.DataFrame(data)

        # Encode categorical variables
        df_encoded = pd.get_dummies(df, columns=["method"])

        # Train model
        X = df_encoded.drop("speedup", axis=1)
        y = df_encoded["speedup"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        return model, df_encoded.columns.tolist()

    def save_results(self, results: List[BenchmarkResult], model_data: Dict):
        """Save benchmark results and model data."""
        # Save raw results
        results_file = self.results_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2, cls=NumpyEncoder)

        # Save model data
        model_file = self.results_dir / "speedup_model.json"
        with open(model_file, "w") as f:
            json.dump(model_data, f, indent=2, cls=NumpyEncoder)

    def run_complete_benchmark(self):
        """Run complete benchmarking pipeline."""
        self.logger.info("Starting benchmark run")

        # Run benchmarks
        results = self.run_benchmarks()
        self.logger.info(f"Completed {len(results)} benchmark cases")

        # Build and evaluate model
        model, feature_names = self.build_predictive_model(results)

        # Save everything
        model_data = {
            "feature_importance": dict(zip(feature_names, model.feature_importances_)),
            "benchmark_metadata": {
                "max_dims": self.max_dims,
                "min_size": self.min_size,
                "max_size": self.max_size,
                "n_samples": self.n_samples,
                "n_repeats": self.n_repeats,
                "methods": self.methods,
            },
        }

        self.save_results(results, model_data)
        self.logger.info("Benchmark pipeline completed")


def analyze_results(results_dir: str):
    """Analyze benchmark results and provide insights."""
    results_file = Path(results_dir) / "benchmark_results.json"
    model_file = Path(results_dir) / "speedup_model.json"

    with open(results_file) as f:
        results = pd.DataFrame(json.load(f))

    with open(model_file) as f:
        model_data = json.load(f)

    # Print summary statistics
    print("\nBenchmark Summary:")
    print(f"Total cases tested: {len(results)}")
    print(f"Average speedup: {results['speedup'].mean():.2f}x")
    print(f"Median speedup: {results['speedup'].median():.2f}x")

    # Print feature importance
    print("\nFeature Importance:")
    for feature, importance in sorted(
        model_data["feature_importance"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature}: {importance:.3f}")

    # Generate recommendations
    print("\nRecommendations:")
    fast_cases = results[results["speedup"] > 1.5]
    if len(fast_cases) > 0:
        print("FastStats is recommended for:")
        for method in fast_cases["method"].unique():
            method_cases = fast_cases[fast_cases["method"] == method]
            print(f"\n{method}:")
            print(f"- Typical speedup: {method_cases['speedup'].median():.2f}x")
            print(
                f"- Best with elements > {method_cases['total_elements'].median():.0f}"
            )
            print(
                f"- Best with reduction ratio > {method_cases['reduction_ratio'].median():.2f}"
            )


if __name__ == "__main__":
    # Example usage
    benchmarker = SmartBenchmarker(
        max_dims=3,
        max_gb=16,
        shape_log_base=10,
        shape_power_range=(0, 4),
        shape_power_step=1,
        n_repeats=5,
        methods=["mean", "median", "std"],
    )

    benchmarker.run_complete_benchmark()
    analyze_results("benchmark_results")
