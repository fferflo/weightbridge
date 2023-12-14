
import numpy as np
import time, types
from collections import defaultdict
from . import tree
from .matchers.base import State, Match
from . import matchers
from functools import partial

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



def adapt(in_values, out_values, in_format=None, out_format=None, in_separator=None, out_separator=None, hints=[], verbose=False):
    t0 = time.time()
    if isinstance(out_values, dict):
        single_input = True
        out_values = [out_values]
    if isinstance(in_values, dict):
        in_values = [in_values]

    # Get separator in input values
    if in_separator is None:
        keys = set()
        def recurse(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    keys.add(k)
                    recurse(v)
        for d in in_values:
            recurse(d)
        in_separator = get_separator_for(keys, "in_separator")
        if in_separator is None:
            in_separator = possible_separators[0]

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

    # Flatten input values
    flat_in_values = {}
    def recurse(source, key=()):
        if isinstance(source, dict):
            for k, v in source.items():
                recurse(v, key + (k,))
        else:
            if len(key) == 0:
                raise ValueError("Input values must be a dictionary")
            name = in_separator.join(key)
            if name in flat_in_values:
                raise ValueError(f"Duplicate input name {name}")
            flat_in_values[name] = np.asarray(source)
    for d in in_values:
        recurse(d)
    in_values = flat_in_values

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
        in_format=in_format,
        out_format=out_format,
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
        partial(matchers.match_by_structured_shapes.match_by_structured_shapes, use_product=True),
        matchers.match_by_paired_parents.match_by_paired_parents,
        matchers.match_by_paired_children.match_by_paired_children,
        matchers.match_unique_leafs.match_unique_leafs,
        matchers.match_number_regex.match_number_regex,
        matchers.match_known_paired_number_regex.match_known_paired_number_regex,
        matchers.match_equivalent_hardcoded_leafs.match_equivalent_hardcoded_leafs,
        matchers.match_passed_hints.match_passed_hints,
        matchers.match_by_paired_prefixes.match_by_paired_prefixes,
        matchers.match_by_paired_predecessors.match_by_paired_predecessors,
    ]
    ops = {op.__name__ if not isinstance(op, partial) else op.func.__name__: op for op in ops}

    # Run matching heuristics
    times = defaultdict(lambda: 0.0)
    changed = True
    while changed:
        changed = False
        for name, op in ops.items():
            if verbose:
                print(f"OP: Trying {name}")
            start = time.time()
            matches, changed = op(state, matches)

            times[name] += time.time() - start
            if changed:
                if verbose:
                    print(f"OP: Changed by {name}")
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
            out_name, out_shape, dtype = x
            in_value = out_values_by_name[LOAD_PREFIX + out_separator + out_name].other.value
            in_name = out_values_by_name[LOAD_PREFIX + out_separator + out_name].other.name

            assert np.prod(in_value.shape) == np.prod(out_shape)

            out_value = state.adapt_format(in_value, out_shape, in_name, out_name)

            return out_value

    out_values = [recurse(treedef) for treedef in out_treedefs]

    if verbose:
        print(f"Matching took {time.time() - t0:.2f} seconds")

    return out_values[0] if single_input else out_values
