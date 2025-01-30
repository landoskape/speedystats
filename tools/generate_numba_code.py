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
def get_{np_method}(data: np.ndarray, keep_axes: Tuple[int]{q_param}) -> np.ndarray:
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

    complete_code = """from typing import Tuple
import numba as nb
import numpy as np\n\n"""

    complete_code += generate_numba_lookup(np_method, max_dims, has_q_param)
    complete_code += "\n".join(all_functions)

    return complete_code


def generate_faststats_module(config):
    # Imports
    template = """
from typing import Union, Iterable, Optional
import numpy as np
from .routing import faststats_route, get_max_dims, get_keep_axes
"""

    # This global variable is used to determine the maximum number of dimensions
    # supported by the fast implementations
    template += f"""MAX_DIMS = get_max_dims()\n\n"""

    # Define the faststats call function
    template += f"""
def _call_faststats(
    data: np.ndarray,
    method: str,
    axis: Union[int, Iterable[int]] = -1,
    keepdims: bool = False,
    q: Optional[float] = None,
) -> np.ndarray:

    # Identify the shape of the data and the axes to keep
    data_ndims = data.ndim
    data_shape = data.shape
    keep_axes = get_keep_axes(axis, data_ndims)

    # If the number of axes to keep is greater than the max supported, use the numpy method directly
    if any(k >= MAX_DIMS for k in keep_axes):
        np_method = getattr(np, method)
        has_q_param = faststats_route(method)[1]
        if has_q_param:
            return np_method(data, axis, keepdims, q)
        else:
            return np_method(data, axis, keepdims)

    # Reshape the data to be flattened along reducing axes
    last_axis = keep_axes[-1]
    if data_ndims > last_axis + 1:
        new_shape = data_shape[: last_axis + 1] + (-1,)
        data = np.reshape(data, new_shape)

    # Get the numba implementation and check if it has a q parameter
    func, has_q_param = faststats_route(method)

    # Call the numba implementation
    if has_q_param:
        out = func(data, keep_axes, q)
    else:
        out = func(data, keep_axes)

    # Reshape the output to match the original data shape if keepdims is True
    if keepdims:
        out = np.expand_dims(out, axis)

    return out
"""

    # Then, for every method in the config, add a function so the user can just call
    # that directly -- and it will use _call_faststats internally for dispatching
    for method_name in config["methods"]:
        if config["methods"][method_name]["has_q_param"]:
            q_signature = ", q: Optional[float] = None"
            q_call = ", q"
        else:
            q_signature = ""
            q_call = ""
        template += f"""
def {method_name}(data: np.ndarray, axis: Union[int, Iterable[int]] = -1, keepdims: bool = False{q_signature}) -> np.ndarray:
    return _call_faststats(data, "{method_name}", axis, keepdims{q_call})
"""
        if config["methods"][method_name]["has_nan_variant"]:
            nan_name = f"nan{method_name}"
            template += f"""
def {nan_name}(data: np.ndarray, axis: Union[int, Iterable[int]] = -1, keepdims: bool = False{q_signature}) -> np.ndarray:
    return _call_faststats(data, "{nan_name}", axis, keepdims{q_call})
"""


def generate_routing_module(config):
    """Generate the routing module that maps numpy methods to their numba implementations."""
    template = """from typing import Callable, Union, Iterable, Tuple
from . import numba\n\n
"""

    # Add faststats_route function
    template += """def faststats_route(np_method: str) -> Callable:
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

    # Add has_q_param logic
    template += "    has_q_param = {\n"
    for method_name in config["methods"]:
        template += f'        "{method_name}": {config["methods"][method_name]["has_q_param"]},\n'
        # Add nan variant if it exists
        if config["methods"][method_name]["has_nan_variant"]:
            nan_name = f"nan{method_name}"
            template += f'        "{nan_name}": {config["methods"][method_name]["has_q_param"]},\n'
    template += "    }\n"

    # Add raise and return logic
    template += """    
    if np_method not in method_map:
        raise ValueError(f"No fast implementation available for {np_method}")
    if np_method not in has_q_param:
        raise ValueError(f"No q param config available for {np_method}")
    return method_map[np_method], has_q_param[np_method]
"""

    # Add get_max_dims function
    template += f"""
def get_max_dims() -> int:
    \"\"\"Get the maximum number of dimensions supported by the fast implementations.
    
    Returns:
        int: Maximum number of dimensions supported
    \"\"\"
    return {config["meta"]["max_dimensions"]}
"""

    # Add get_keep_axes function
    template += f"""
def get_keep_axes(axis: Union[int, Iterable[int]], ndim: int) -> Tuple[int]:
    keep_axes = list(range(ndim))
    if isinstance(axis, int):
        axis = [axis]
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

    # Generate and write the routing module
    routing_code = generate_routing_module(config)
    with open(route_path, "w") as f:
        f.write(routing_code)
    print(f"Generated {route_path}")

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
