"""
Template for generating Numba functions.

This is a template for generating Numba functions. It is used to generate the
functions for the mean, std, and other functions that are used in the faststats
package. The template produces a function that computes a statistic from the 
numpy library using numba speedups using (nested) loops. 

An example of the template is shown below, we'll use it to discuss what's 
happening in the template generator function.
```python
@nb.njit(parallel=True, fastmath=True, cache=True)
def numba_mean_keep01(data):
    '''Numba speedup for mean reducing all but axes (0, 1)'''
    output = np.zeros((data.shape[0], date.shape[1]))
    for n0 in nb.prange(data.shape[0]):
        for n1 in nb.prange(data.shape[1]):
            output[n0, n1] = np.mean(data[n0, n1])
    return output
```

Each numba function is compiled with nopython, parallel, fastmath, and cache.
The function name indicates the statistic being computed and the axes that are
_kept_ (not reduced!). This is important because it works for a numpy array 
with any number of dimensions (as long as it's at least as many as are kept!).
"""

import os
from itertools import combinations
import yaml


def generate_numba_function(np_method, keep_axes, fastmath, parallel, cache):
    """
    Generate a Numba function that computes mean while keeping specified axes.

    Args:
        np_method: str, the name of the numpy method to use
        keep_axes: int or tuple of ints representing axes to keep

    Returns:
        str: The generated function code as a string
    """
    raise ValueError("it doesn't work yet.....")
    if isinstance(keep_axes, int):
        keep_axes = (keep_axes,)

    max_axis = max(keep_axes)

    # Create function name
    func_name = f"numba_{np_method}_keep{''.join(str(ax) for ax in keep_axes)}"

    # Generate the output shape calculation
    shape_dims = [f"data.shape[{i}]" for i in keep_axes]
    output_shape = ", ".join(shape_dims)

    # Generate the nested loops
    indent = "    "
    loop_vars = [f"n{i}" for i in range(len(keep_axes))]
    loop_ranges = [f"nb.prange(data.shape[{ax}])" for ax in keep_axes]

    loops = ""
    for var, range_expr in zip(loop_vars, loop_ranges):
        loops += f"{indent}for {var} in {range_expr}:\n"
        indent += "    "

    # Generate the indexing for the output and input arrays
    out_index = ", ".join(loop_vars)

    # Generate the data indexing
    data_index_parts = []
    for i in range(max(1, max_axis)):
        if i in keep_axes:
            idx = loop_vars[keep_axes.index(i)]
            data_index_parts.append(idx)
        else:
            data_index_parts.append(":")

    # Join data part indices
    data_index = ", ".join(data_index_parts)

    # Create the function template
    template = f'''
@nb.njit(parallel={parallel}, fastmath={fastmath}, cache={cache})
def {func_name}(data: np.ndarray) -> np.ndarray:
    """Numba speedup for {np_method} reducing all but axes {keep_axes}"""
    output = np.zeros(({output_shape}))
{loops}{indent}output[{out_index}] = np.{np_method}(data[{data_index}])
    return output
'''

    return template


def generate_function_lookup(np_method, max_dims):
    """
    def get_mean(data, axis):
        keep_axes = list(range(data.ndim))
        if not hasattr(axis, "__iter__"):
            axis = [axis]
        for a in axis:
            keep_axes.remove(a)

        keep_axes = tuple(keep_axes)
        if keep_axes == (0,):
            return numba_mean_keep0(data)
        elif keep_axes == (1,):
            return numba_mean_keep1(data)
        elif keep_axes == (2,):
            return numba_mean_keep2(data)
        elif keep_axes == (0, 1):
            return numba_mean_keep01(data)
        elif keep_axes == (0, 2):
            return numba_mean_keep02(data)
        elif keep_axes == (1, 2):
            return numba_mean_keep12(data)
        else:
            raise ValueError("Invalid data shape")
    """


def get_all_combinations(max_dims):
    """
    Get all combinations of axes for a given number of dimensions.

    Excludes single axes and all axes.

    Args:
        max_dims: maximum number of dimensions to consider

    Returns:
        list: all combinations of axes
    """
    axis_combinations = []
    for dim in range(2, max_dims + 1):
        for axes in combinations(range(dim), dim - 1):
            axis_combinations.append(axes)
    return axis_combinations


def generate_module(np_method, max_dims, fastmath, parallel, cache):
    """
    Generate a module containing all possible numba functions up to max_dims.

    Args:
        np_method: str, the name of the numpy method to use
        max_dims: maximum number of dimensions to consider

    Returns:
        str: Complete code containing all generated functions
    """
    axis_combinations = get_all_combinations(max_dims)

    all_functions = []
    for comb in axis_combinations:
        all_functions.append(
            generate_numba_function(
                np_method,
                comb,
                fastmath,
                parallel,
                cache,
            )
        )

    complete_code = """import numba as nb
import numpy as np
"""
    complete_code += "\n".join(all_functions)

    return complete_code


# Example usage:
if __name__ == "__main__":
    # Load configuration
    with open("tools/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Get configuration parameters
    max_dims = config["meta"]["max_dimensions"]
    parallel = config["meta"]["parallel"]
    cache = config["meta"]["cache"]
    output_path = config["meta"]["output_path"]

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Generate code for each method
    for method_name, method_config in config["methods"].items():
        print(method_name)

        if method_config["has_q_param"]:
            continue

        numpy_name = method_config["numpy_name"]
        code = generate_module(
            np_method=numpy_name,
            max_dims=max_dims,
            fastmath=method_config["fastmath"],
            parallel=parallel,
            cache=cache,
        )

        # Write code to output file
        output_file = os.path.join(output_path, f"{numpy_name}.py")
        with open(output_file, "w") as f:
            f.write(code)
        print(f"Generated {output_file}")

        # If the method has a nanvariant, add it
        if method_config["has_nan_variant"]:
            numpy_name = f"nan{numpy_name}"
            code = generate_module(
                np_method=numpy_name,
                max_dims=max_dims,
                fastmath=method_config["fastmath"],
                parallel=parallel,
                cache=cache,
            )
            output_file = os.path.join(output_path, f"{numpy_name}.py")
            with open(output_file, "w") as f:
                f.write(code)
            print(f"Generated {output_file}")
