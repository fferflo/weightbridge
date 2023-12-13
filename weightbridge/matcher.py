
import numpy as np
import time
from collections import defaultdict
from . import tree
from .matchers.base import State, Match
from . import matchers

LOAD_PREFIX = "load_prefix"

possible_separators = ["/", "."]
def get_separator_for(names, type):
    has_separator = [any(s in name for name in names) for s in possible_separators]
    n = np.count_nonzero(has_separator)
    if n == 0:
        return None
    elif n == 1:
        return possible_separators[np.argmax(has_separator, axis=0)]
    else:
        raise ValueError(f"Could not implicitly determine separator in weight names. Please provide argument {type}.")

format_map = {
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "flax": "tensorflow",
    "haiku": "tensorflow",
}

def adapt_format(value, out_shape, in_name, out_name, in_format, out_format):
    if (in_format is None) != (out_format is None):
        raise ValueError("Either both or neither of in_format and out_format must be specified")
    if in_format is not None:
        if not in_format in format_map:
            raise ValueError(f"Format {in_format} not supported")
        if not out_format in format_map:
            raise ValueError(f"Format {out_format} not supported")
        in_format = format_map[in_format]
        out_format = format_map[out_format]
        if in_format != out_format:
            if in_format == "pytorch" and out_format == "tensorflow":
                if (in_name.endswith(".weight") or in_name == "weight") and len(out_shape) >= 2 or (in_name.endswith(".in_proj_weight") and len(out_shape) == 2):
                    perm = list(range(2, len(out_shape))) + [1, 0]
                    perm_inv = [perm.index(i) for i in range(len(perm))]
                    in_shape = [out_shape[i] for i in perm_inv]

                    if value.shape != in_shape:
                        value = np.reshape(value, in_shape)
                    value = np.transpose(value, perm)
                    assert value.shape == out_shape, f"{value.shape} != {out_shape}"
            elif in_format == "tensorflow" and out_format == "pytorch":
                print("weightbridge - warning: Tensorflow to PyTorch conversion is not tested") # TODO: test this
                if (out_name.endswith(".weight") or out_name == "weight") and len(out_shape) >= 2 or (out_name.endswith(".in_proj_weight") and len(out_shape) == 2):
                    perm = [len(out_shape) - 1, len(out_shape) - 2] + list(range(len(out_shape) - 2))
                    perm_inv = [perm.index(i) for i in range(len(perm))]
                    in_shape = [out_shape[i] for i in perm_inv]

                    if value.shape != in_shape:
                        value = np.reshape(value, in_shape)
                    value = np.transpose(value, perm)
                    assert value.shape == out_shape, f"{value.shape} != {out_shape}"
            else:
                raise ValueError(f"Conversion from {in_format} to {out_format} not supported")
    
    if value.shape != out_shape:
        value = np.reshape(value, out_shape)
    return value

def match(in_values, out_values, in_format=None, out_format=None, in_separator=None, out_separator=None, hints=[], verbose=False):
    t0 = time.time()
    if isinstance(out_values, dict):
        single_input = True
        out_values = [out_values]
    in_values = {k: np.asarray(v) for k, v in in_values.items()}

    # Get separator in input values
    if in_separator is None:
        in_separator = get_separator_for(in_values.keys(), "in_separator")

    # Get separator in output values
    if out_separator is None:
        keys = set()
        def recurse(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    keys.add(k)
                    recurse(v)
        for d in out_values:
            recurse(d)
        out_separator = get_separator_for(keys, "out_separator")
        if out_separator is None:
            out_separator = possible_separators[0]

    # Flatten output values and remember tree structure
    flat_out_values = {}
    def recurse(source, key=()):
        if isinstance(source, dict):
            treedef = {}
            for k, v in source.items():
                treedef[k] = recurse(v, key + (k,))
            return treedef
        else:
            if len(key) == 0:
                raise ValueError("Output values must be a dictionary")
            name = out_separator.join(key)
            if name in flat_out_values:
                raise ValueError(f"Duplicate output name {name}")
            source = np.asarray(source)
            flat_out_values[name] = source
            return (name, source.shape, source.dtype)
    out_treedefs = []
    for d in out_values:
        out_treedefs.append(recurse(d))
    out_values = flat_out_values



    state = State(
        in_values,
        {LOAD_PREFIX + out_separator + k: v for k, v in out_values.items()},
        in_separator=in_separator,
        out_separator=out_separator,
        hints=hints,
        verbose=verbose,
    )

    # Build module trees
    out_tree = tree.build(state.out_values)
    in_tree = tree.build(state.in_values)

    # Initialize matches
    matches = [Match(list(in_tree), list(out_tree), "Root")]

    # Construct heuristics
    ops = [
        matchers.match_by_structured_shapes.match_by_structured_shapes,
        matchers.match_by_paired_parents.match_by_paired_parents,
        matchers.match_by_paired_children.match_by_paired_children,
        matchers.match_unique_leafs.match_unique_leafs,
        matchers.match_number_regex.match_number_regex,
        matchers.match_known_paired_number_regex.match_known_paired_number_regex,
        matchers.match_equivalent_hardcoded_leafs.match_equivalent_hardcoded_leafs,
        matchers.match_passed_hints.match_passed_hints,
        matchers.match_subnames.match_subnames,
        matchers.match_by_paired_predecessors.match_by_paired_predecessors,
        matchers.match_by_paired_prefixes.match_by_paired_prefixes,
    ]

    # Run matching heuristics
    times = defaultdict(lambda: 0.0)
    changed = True
    while changed:
        changed = False
        for op in ops:
            if verbose:
                print(f"OP: Trying {op.__name__}")
            start = time.time()
            matches, changed = op(state, matches)

            times[op.__name__] += time.time() - start
            if changed:
                if verbose:
                    print(f"OP: Changed by {op.__name__}")
                break

    if verbose:
        print("Times (sec) per operation:")
        for k, v in sorted(times.items(), key=lambda t: t[1]):
            print(f"    {k} {v}")

    if len(matches) > 0:
        print()
        for match in matches:
            print("Failed to pair the following nodes")
            for n in match.out_nodes:
                print(f"    OUT {n.full_prefix} {n.get_structured_shapes()}")
            for n in match.in_nodes:
                print(f"    IN  {n.full_prefix} {n.get_structured_shapes()}")
        raise ValueError("Failed to pair input values with output values")

    if verbose:
        print()
        print("Paired values:")
        mapping = {out_node.full_prefix: in_node.full_prefix for out_node, in_node in state.paired_nodes if out_node.is_leaf() and in_node.is_leaf()}
        for out_value in state.out_values:
            print(f"    {out_value.name} -> {mapping[out_value.name]}")

    # Matching successful! Now build tree of output values
    out_values_by_name = {v.name: v for v in state.out_values}
    def recurse(x):
        if isinstance(x, dict):
            return {k: recurse(v) for k, v in x.items()}
        else:
            out_name, shape, dtype = x
            value = out_values_by_name[LOAD_PREFIX + out_separator + out_name].other.value
            in_name = out_values_by_name[LOAD_PREFIX + out_separator + out_name].other.name

            if value.dtype != dtype:
                value = value.astype(dtype)
            assert np.prod(value.shape) == np.prod(shape)

            value = adapt_format(value, shape, in_name, out_name, in_format, out_format)

            return value

    out_values = [recurse(treedef) for treedef in out_treedefs]

    if verbose:
        print(f"Matching took {time.time() - t0:.2f} seconds")

    return out_values[0] if single_input else out_values
