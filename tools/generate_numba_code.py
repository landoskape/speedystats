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


def get_func_name(np_method, keep_axes):
    return f"numba_{np_method}_keep{''.join(str(ax) for ax in keep_axes)}"


def generate_numba_function(
    np_method: str,
    keep_axes: tuple,
    fastmath: bool,
    parallel: bool,
    cache: bool,
    has_q_param: bool,
) -> str:
    """
    Generate a Numba function that computes mean while keeping specified axes.

    Args:
        np_method: str, the name of the numpy method to use
        keep_axes: int or tuple of ints representing axes to keep

    Returns:
        str: The generated function code as a string
    """
    max_axis = max(keep_axes)

    # Create function name
    func_name = get_func_name(np_method, keep_axes)

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
    for i in range(max_axis + 1):
        if i in keep_axes:
            idx = loop_vars[keep_axes.index(i)]
            data_index_parts.append(idx)
        else:
            data_index_parts.append(":")

    # Join data part indices
    data_index = ", ".join(data_index_parts)

    # Create the function template
    q_param = ", q" if has_q_param else ""
    template = f'''
@nb.njit(parallel={parallel}, fastmath={fastmath}, cache={cache})
def {func_name}(data: np.ndarray{q_param}) -> np.ndarray:
    """Numba speedup for {np_method} reducing all but axes {keep_axes}"""
    output = np.zeros(({output_shape}))
{loops}{indent}output[{out_index}] = np.{np_method}(data[{data_index}{q_param}])
    return output
'''

    return template


def lookup_template(np_method, has_q_param):
    q_param = ", q" if has_q_param else ""
    return f"""
def get_{np_method}(data, axis{q_param}):
    keep_axes = get_keep_axes(axis, data.ndim)
"""


def generate_numba_lookup(np_method, max_dims, has_q_param):
    axis_combinations = get_all_combinations(max_dims)

    template = lookup_template(np_method, has_q_param)
    for keep_axes in axis_combinations:
        q_param = ", q" if has_q_param else ""
        func_name = get_func_name(np_method, keep_axes)
        template += f"    if keep_axes == {keep_axes}:\n"
        template += f"        return {func_name}(data{q_param})\n"

    template += f"    raise ValueError(f'Invalid data shape for {np_method}')\n"

    return template


def get_all_combinations(max_dims):
    """
    Get all combinations of axes for a given number of dimensions.

    Excludes single axes and all axes.

    Args:
        max_dims: maximum number of dimensions to consider

    Returns:
        list: all combinations of axes
    """

    def _make_tuple(keep_axes):
        if isinstance(keep_axes, int):
            keep_axes = (keep_axes,)
        return keep_axes

    axis_combinations = []
    for dim in range(2, max_dims + 1):
        for axes in combinations(range(dim), dim - 1):
            axis_combinations.append(_make_tuple(axes))
    return axis_combinations


def generate_module(np_method, max_dims, fastmath, parallel, cache, has_q_param):
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
                has_q_param,
            )
        )

    complete_code = """import numba as nb
import numpy as np

from ..routing import get_keep_axes
"""
    complete_code += generate_numba_lookup(np_method, max_dims, has_q_param)
    complete_code += "\n".join(all_functions)

    return complete_code


def generate_routing_module(config):
    """Generate the routing module that maps numpy methods to their numba implementations."""
    template = """from typing import Callable
from . import numba

def faststats_route(np_method: str) -> Callable:
    \"\"\"Route numpy method names to their fast implementations.
    
    Args:
        np_method: Name of the numpy method to route
        
    Returns:
        Callable: The corresponding fast implementation
    \"\"\"
"""
    # Add routing logic for each method
    template += "    method_map = {\n"

    for method_name in config["methods"]:
        template += f'        "{method_name}": numba.{method_name}.get_{method_name},\n'

        # Add nan variant if it exists
        if config["methods"][method_name]["has_nan_variant"]:
            nan_name = f"nan{method_name}"
            template += f'        "{nan_name}": numba.{nan_name}.get_{nan_name},\n'

    template += "    }\n"
    template += """    
    if np_method not in method_map:
        raise ValueError(f"No fast implementation available for {np_method}")
    return method_map[np_method]
"""

    template += f"""
def get_max_dims() -> int:
    \"\"\"Get the maximum number of dimensions supported by the fast implementations.
    
    Returns:
        int: Maximum number of dimensions supported
    \"\"\"
    return {config["meta"]["max_dimensions"]}
"""

    template += f"""
def get_keep_axes(axis, ndim):
    keep_axes = list(range(ndim))
    for a in axis:
        keep_axes.remove(a)

    keep_axes = tuple(keep_axes)
    return keep_axes
"""
    return template


def generate_init_file(config):
    """Generate the __init__.py file that imports all get_* methods."""
    template = """\"\"\"Numba-accelerated statistical functions.\"\"\"

"""
    # Add imports for each method
    for method_name in config["methods"]:
        template += f"from .{method_name} import get_{method_name}\n"

        # Add nan variant if it exists
        if config["methods"][method_name]["has_nan_variant"]:
            nan_name = f"nan{method_name}"
            template += f"from .{nan_name} import get_{nan_name}\n"

    return template


# Example usage:
if __name__ == "__main__":
    # Load configuration
    with open("tools/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Get configuration parameters
    max_dims = config["meta"]["max_dimensions"]
    parallel = config["meta"]["parallel"]
    cache = config["meta"]["cache"]
    numba_path = config["meta"]["output_path"] + "/numba"
    init_path = config["meta"]["output_path"] + "/__init__.py"
    route_path = config["meta"]["output_path"] + "/routing.py"

    # Ensure output directory exists
    os.makedirs(numba_path, exist_ok=True)

    # Generate and write the __init__.py file
    init_code = generate_init_file(config)
    with open(os.path.join(numba_path, "__init__.py"), "w") as f:
        f.write(init_code)
    print(f"Generated {numba_path}/__init__.py")

    # Generate code for each method
    for method_name in config["methods"]:
        code = generate_module(
            np_method=method_name,
            max_dims=max_dims,
            fastmath=config["methods"][method_name]["fastmath"],
            parallel=parallel,
            cache=cache,
            has_q_param=config["methods"][method_name]["has_q_param"],
        )

        # Write code to output file
        output_file = os.path.join(numba_path, f"{method_name}.py")
        with open(output_file, "w") as f:
            f.write(code)
        print(f"Generated {output_file}")

        # If the method has a nanvariant, add it
        if config["methods"][method_name]["has_nan_variant"]:
            nan_name = f"nan{method_name}"
            code = generate_module(
                np_method=nan_name,
                max_dims=max_dims,
                fastmath=config["methods"][method_name]["fastmath"],
                parallel=parallel,
                cache=cache,
                has_q_param=config["methods"][method_name]["has_q_param"],
            )
            output_file = os.path.join(numba_path, f"{nan_name}.py")
            with open(output_file, "w") as f:
                f.write(code)
            print(f"Generated {output_file}")

    # Generate and write the routing module
    routing_code = generate_routing_module(config)
    with open(route_path, "w") as f:
        f.write(routing_code)
    print(f"Generated {route_path}")
