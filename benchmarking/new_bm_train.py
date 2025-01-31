from typing import Tuple
from tqdm import tqdm
import numpy as np
from itertools import combinations, product
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import faststats


def generate_shapes(
    max_dims: int = 3,
    shape_log_base: int = 10,
    shape_power_range: Tuple[int, int] = (0, 4),
    shape_power_step: int = 1,
    max_gb: int = 16,
):
    """
    Generate array shapes using logarithmic sampling to ensure good coverage
    across orders of magnitude.
    """
    bytes_per_element = 8
    max_bytes = max_gb * 1024**3
    shapes = []
    array_nbytes = []
    dim_shapes = shape_log_base ** np.arange(
        shape_power_range[0],
        shape_power_range[1] + shape_power_step,
        shape_power_step,
    )
    dim_shapes = [int(x) for x in dim_shapes]

    # Generate different dimensionalities
    for ndim in range(2, max_dims + 1):
        options = [dim_shapes] * ndim
        for shape in product(*options):
            if np.prod(shape) * bytes_per_element <= max_bytes:
                shapes.append(shape)
                array_nbytes.append(np.prod(shape) * bytes_per_element)

    # Sort first by dimension count, then by array size
    shapes = sorted(shapes, key=lambda x: (len(x), array_nbytes[shapes.index(x)]))
    return shapes


def get_axes_combinations(ndim):
    """Get all valid reduction axis combinations"""
    all_axes = []
    for r in range(1, ndim):  # Don't include reducing all axes
        all_axes.extend(combinations(range(ndim), r))
    return all_axes


def run_benchmark(data, axis, method="mean"):
    """Run benchmark comparing numpy vs faststats"""
    numpy_func = getattr(np, method)
    faststats_func = getattr(faststats, method)

    # Warm-up run with small data array
    ndims = data.ndim
    warmup_data = np.random.randn(*[5] * ndims)
    _ = numpy_func(warmup_data, axis=axis)
    _ = faststats_func(warmup_data, axis=axis)

    # Time NumPy
    start = time.perf_counter()
    _ = numpy_func(data, axis=axis)
    numpy_time = time.perf_counter() - start

    # Time FastStats
    start = time.perf_counter()
    _ = faststats_func(data, axis=axis)
    faststats_time = time.perf_counter() - start

    return numpy_time, faststats_time


def generate_features(shape, axis):
    """Generate features for shape/axis combination"""
    features = {
        "ndim": len(shape),
        "total_size": np.prod(shape),
        "max_dim": max(shape),
        "min_dim": min(shape),
        "mean_dim": np.mean(shape),
        "std_dim": np.std(shape),
        "num_axes_reduced": len(axis) if isinstance(axis, tuple) else 1,
        "max_reduced_dim": (
            max(shape[i] for i in axis) if isinstance(axis, tuple) else shape[axis]
        ),
        "min_reduced_dim": (
            min(shape[i] for i in axis) if isinstance(axis, tuple) else shape[axis]
        ),
        "reduced_size": (
            np.prod([shape[i] for i in axis])
            if isinstance(axis, tuple)
            else shape[axis]
        ),
        "kept_size": np.prod([shape[i] for i in range(len(shape)) if i not in axis]),
        "shape_str": str(shape),  # Keep original shape for reference
        "axis_str": str(axis),  # Keep original axis for reference
    }

    # Add log-transformed features
    for key in ["total_size", "max_dim", "min_dim", "reduced_size", "kept_size"]:
        features[f"log_{key}"] = np.log10(features[key])

    return features


def create_training_dataset(shapes, method="mean", n_repeats=3):
    """Create training dataset from shapes and axes"""
    data = []

    shape_progress = tqdm(shapes, desc="Processing shapes")
    for shape in shape_progress:
        shape_progress.set_description(f"Processing shape {shape}")
        axes = get_axes_combinations(len(shape))

        # Create random array once for each shape
        arr = np.random.randn(*shape)

        for axis in axes:
            features = generate_features(shape, axis)

            # Run multiple times and take mean
            speedups = []
            for _ in range(n_repeats):
                numpy_time, faststats_time = run_benchmark(arr, axis, method)
                speedups.append(numpy_time / faststats_time)

            features["speedup"] = np.mean(speedups)
            data.append(features)

    return pd.DataFrame(data)


def train_model(df, test_size=0.2):
    """Train RandomForestRegressor on the dataset"""
    # Separate features used for training
    feature_cols = [
        "ndim",
        "log_total_size",
        "log_max_dim",
        "log_min_dim",
        "mean_dim",
        "std_dim",
        "num_axes_reduced",
        "log_reduced_size",
        "log_kept_size",
    ]

    X = df[feature_cols]
    y = df["speedup"]

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"Train R² score: {train_score:.3f}")
    print(f"Test R² score: {test_score:.3f}")

    # Save model and scaler
    joblib.dump(model, "speedup_model.joblib")
    joblib.dump(scaler, "speedup_scaler.joblib")

    return model, scaler, feature_cols


def predict_speedup(shape, axis, model, scaler, feature_cols):
    """Predict speedup for new shape/axis combination"""
    features = generate_features(shape, axis)
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]


if __name__ == "__main__":
    # Generate shapes
    shapes = generate_shapes(max_dims=3, shape_power_range=(0, 3))

    # Create dataset
    df = create_training_dataset(shapes, method="mean", n_repeats=3)

    # Train model
    model, scaler, feature_cols = train_model(df)

    # Example prediction
    test_shape = (3000, 400, 807)
    test_axis = (0, 1)
    predicted_speedup = predict_speedup(
        test_shape, test_axis, model, scaler, feature_cols
    )
    numpy_speed, faststats_speed = run_benchmark(
        np.random.randn(*test_shape), test_axis, method="mean"
    )
    true_speedup = numpy_speed / faststats_speed
    print(f"\nExample prediction:")
    print(f"Shape: {test_shape}, Axis: {test_axis}")
    print(f"Predicted speedup: {predicted_speedup:.2f}x")
    print(f"True speedup: {true_speedup:.2f}x\n")

    # Test a few more cases
    test_cases = [
        ((1000, 1000), (0,)),
        ((100, 100, 100), (0, 1)),
        ((10000, 100), (1,)),
        ((500, 500, 500), (0, 2)),
    ]

    print("\nAdditional test cases:")
    for shape, axis in test_cases:
        speedup = predict_speedup(shape, axis, model, scaler, feature_cols)
        numpy_speed, faststats_speed = run_benchmark(
            np.random.randn(*shape), axis, method="mean"
        )
        true_speedup = numpy_speed / faststats_speed
        print(f"Shape: {shape}, Axis: {axis}")
        print(f"Predicted speedup: {speedup:.2f}x")
        print(f"True speedup: {true_speedup:.2f}x\n")
