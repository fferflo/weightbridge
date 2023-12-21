
import numpy as np
import time, types, traceback, os, json, sys
from collections import defaultdict
from . import tree
from .matchers.base import Match
from . import matchers
from functools import partial
from .state import State

LOAD_PREFIX = "load_prefix"
CACHE_VERSION = 1

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


def _cache_key(x):
    return [[k, list(v.shape)] for k, v in x.items()]

def adapt(in_values, out_values, in_format=None, out_format=None, in_separator=None, out_separator=None, hints=[], cache=None, ignore_unmatched_inputs=False, ignore_unmatched_outputs=False, verbose=False):
    """Adapt weights in ``in_values`` to match the signature of ``out_values``.

    Args:
        in_values: The weights that will be loaded. A dictionary from weight name to weight value (e.g. Numpy or Torch tensor). E.g.
            result of ``torch.load(...)``.
        out_values: The original (random) weights of the model into which weights will be loaded. A dictionary from weight name to weight
            value (e.g. Numpy or Torch tensor). E.g. result of ``model.state_dict()``.
        in_format: The format defining the order of dimensions in the input weights, e.g. ``"pytorch"`` or ``"tensorflow"``. If
            ``None``, weights are not transposed. Defaults to ``None``.
        out_format: The format defining the order of dimensions in the output weights, e.g. ``"pytorch"`` or ``"tensorflow"``. If
            ``None``, weights are not transposed. Defaults to ``None``.
        in_separator: A string separating different modules in the input weight names (e.g. ``"."`` or ``"/"``). If ``None``, the separator
            is inferred from the input weight names. Defaults to ``None``.
        out_separator: A string separating different modules in the output weight names (e.g. ``"."`` or ``"/"``). If ``None``, the
            separator is inferred from the output weight names. Defaults to ``None``.
        hints: A list of hints that can be used to resolve ambiguous matches between input and output weights. Each hint is a pair of
            strings ``(out_name, in_name)`` where ``out_name`` and ``in_name`` are substrings of output and input weight names
            that uniquely identify the pair as a match. For example, for a matching failure that is reported by weightbridge as
            ```
            Failed to pair the following nodes
                OUT load_prefix/encode/stage3/block6/reduce/linear/w ((262144,),)
                OUT load_prefix/encode/stage3/block6/expand/linear/w ((262144,),)
                IN  backbone.0.body.layer3.5.conv1.weight ((262144,),)
                IN  backbone.0.body.layer3.5.conv3.weight ((262144,),)
            ```
            we can pass ``hints=[("reduce", "conv1")]`` to resolve the ambiguity.
        cache: A path to a cache file where successful matches will be stored and reloaded in subsequent calls. If it is not an absolute
            path, it is interpreted as relative to the file that called ``adapt``. If ``None``, does not use cache. Defaults to ``None``.
        ignore_unmatched_inputs: If ``True``, does not raise an error if some input weights are not matched to output weights. Defaults to
            ``False``.
        ignore_unmatched_outputs: If ``True``, does not raise an error if some output weights are not matched to input weights. Defaults to
            ``False``.
        verbose: If ``True``, prints information on the used matching steps and the final mapping. Defaults to ``False``.

    Returns:
        The adapted weights as a dictionary from weight name to weight value matching the signature of ``out_values``.
    """
    
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
            flat_out_values[name] = source
            return (name, tuple(source.shape), str(source.dtype))
    out_treedefs = []
    for d in out_values:
        out_treedefs.append(recurse(d))
    out_values = flat_out_values

    state = State(
        {LOAD_PREFIX + in_separator + k: v for k, v in in_values.items()},
        {LOAD_PREFIX + out_separator + k: v for k, v in out_values.items()},
        in_separator=in_separator,
        out_separator=out_separator,
        in_format=in_format,
        out_format=out_format,
        hints=hints,
        ignore_unmatched_inputs=ignore_unmatched_inputs,
        ignore_unmatched_outputs=ignore_unmatched_outputs,
        verbose=verbose,
    )

    # Build module trees
    out_tree = tree.build(state.out_values)
    in_tree = tree.build(state.in_values)

    # Check if mapping is cached
    cache_miss = True
    if not cache is None:
        if not cache.endswith(".json"):
            cache += ".json"
        if not os.path.isabs(cache):
            stack = traceback.extract_stack()
            assert len(stack) >= 2
            path, _, _, _ = stack[-2]
            path = os.path.dirname(path)
            cache = os.path.join(path, cache)

        cache_key = [
            CACHE_VERSION,
            _cache_key(in_values),
            _cache_key(out_values),
            in_format,
            out_format,
            in_separator,
            out_separator,
        ]

        if os.path.isfile(cache):
            with open(cache, "r") as f:
                data = json.load(f)
            if not data is None and "key" in data and "mapping" in data and data["key"] == cache_key:
                in_nodes = {node.full_prefix: node for node in in_tree}
                out_nodes = {node.full_prefix: node for node in out_tree}

                for out_name, in_name in data["mapping"]:
                    if not out_name in out_nodes or not in_name in in_nodes:
                        if verbose:
                            print("Found cache file, but it doesnot match the input data. Rerunning matching.")
                        break
                else:
                    if verbose:
                        print("Found cache file, and it matches the input data. Loading mapping.")
                    cache_miss = False
                    for out_name, in_name in data["mapping"]:
                        state.pair_node(in_nodes[in_name], out_nodes[out_name])
            else:
                if verbose:
                    print("Found cache file, but it doesnot match the input data. Rerunning matching.")
        else:
            if verbose:
                print("Found no cache file. Running matching.")

    if cache_miss:
        # Initialize matches
        matches = [Match(list(in_tree), list(out_tree), "Root")]

        # Construct heuristics
        ops = [
            partial(matchers.match_by_structured_shapes.match_by_structured_shapes, format="product"),
            partial(matchers.match_by_structured_shapes.match_by_structured_shapes, format="as-input-shape"),
            partial(matchers.match_by_structured_shapes.match_by_structured_shapes, format="as-output-shape"),
            matchers.match_by_paired_parents.match_by_paired_parents,
            matchers.match_by_paired_children.match_by_paired_children,
            matchers.match_unique_leafs.match_unique_leafs,
            matchers.match_number_regex.match_number_regex,
            matchers.match_identical_names.match_identical_names,
            matchers.match_known_paired_number_regex.match_known_paired_number_regex,
            matchers.match_equivalent_hardcoded_leafs.match_equivalent_hardcoded_leafs,
            matchers.match_passed_hints.match_passed_hints,
            matchers.match_by_paired_prefixes.match_by_paired_prefixes,
            matchers.match_by_paired_predecessors.match_by_paired_predecessors,
        ]
        def op_to_string(op):
            if isinstance(op, partial):
                args = [str(a) for a in op.args] + [f"{k}={v}" for k, v in op.keywords.items()]
                return f"{op.func.__name__}({', '.join(args)})"
            else:
                return op.__name__
        ops = [(op_to_string(op), op) for op in ops]

        # Run matching heuristics
        times = defaultdict(lambda: 0.0)
        changed = True
        while changed:
            changed = False
            for name, op in ops:
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

        # Check if any hints are unused
        unused_hints = set(state.hints) - set(state.used_hints)
        if len(unused_hints) > 0:
            raise ValueError(f"Unused hints: {unused_hints}")

    # Matching successful!

    # Save to cache
    if not cache is None and cache_miss:
        data = {
            "key": cache_key,
            "mapping": [[out_node.full_prefix, in_node.full_prefix] for out_node, in_node in state.paired_nodes if out_node.is_leaf() and in_node.is_leaf()],
        }
        with open(cache, "w") as f:
            json.dump(data, f)

    if len(state.paired_nodes) == 0:
        raise ValueError("No weights were paired")

    if verbose:
        print()
        print("Paired values:")
        mapping = {out_node.full_prefix: in_node for out_node, in_node in state.paired_nodes if out_node.is_leaf() and in_node.is_leaf()}
        for out_value in state.out_values:
            print(f"    {out_value.name} {out_value.shape} -> {mapping[out_value.name].full_prefix} {mapping[out_value.name].value.shape}")
        
        unpaired_in_values = [in_value for in_value in state.in_values if in_value.other is None]
        unpaired_out_values = [out_value for out_value in state.out_values if out_value.other is None]
        if len(unpaired_in_values) > 0 or len(unpaired_out_values) > 0:
            print()
            print("Unpaired leafs:")
            for out_value in unpaired_out_values:
                print(f"    OUT {out_value.name} {out_value.shape}")
            for in_value in unpaired_in_values:
                print(f"    IN  {in_value.name} {in_value.shape}")

    # Create output dicts from stored tree structure
    out_values_by_name = {v.name: v for v in state.out_values}
    def recurse(x):
        if isinstance(x, dict):
            return {k: recurse(v) for k, v in x.items()}
        else:
            out_name, out_shape, dtype = x
            out_value = out_values_by_name[LOAD_PREFIX + out_separator + out_name]
            out_type = type(out_value.value)
            if out_value.other is None:
                assert state.ignore_unmatched_outputs
                out_tensor = out_value.value
            else:
                in_tensor = out_value.other.value
                in_name = out_value.other.name

                assert np.prod(in_tensor.shape) == np.prod(out_shape)

                out_tensor = state.formatter.adapt_format(in_tensor, out_shape, in_name, out_name)

                if "torch" in sys.modules:
                    import torch
                    if out_type == torch.Tensor:
                        out_tensor = torch.as_tensor(out_tensor)

            return out_tensor
    out_values = [recurse(treedef) for treedef in out_treedefs]

    if verbose:
        print(f"weightmatcher.adapt took {time.time() - t0:.2f} seconds")

    return out_values[0] if single_input else out_values
