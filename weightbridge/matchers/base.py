import types
import numpy as np
import weightbridge.tree as tree

class Match:
    def __init__(self, in_nodes, out_nodes, description=None):
        assert isinstance(in_nodes, list) and isinstance(out_nodes, list)
        for n in in_nodes + out_nodes:
            assert isinstance(n, tree.Node)
        assert len(in_nodes) == len(set(id(n) for n in in_nodes))
        assert len(out_nodes) == len(set(id(n) for n in out_nodes))
        self.in_nodes = list(in_nodes)
        self.out_nodes = list(out_nodes)
        self.description = description

    def __sub__(self, other):
        assert isinstance(other, Match)
        return Match(
            [n for n in self.in_nodes if n not in other.in_nodes],
            [n for n in self.out_nodes if n not in other.out_nodes],
        )

format_map = {
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "flax": "tensorflow",
    "haiku": "tensorflow",
    "jittor": "pytorch",
}

def can_reshape(in_shape, out_shape):
    if tuple(s for s in in_shape if s != 1) == tuple(s for s in out_shape if s != 1):
        return True
    in_shape = tuple(s for s in in_shape if s != 1)
    out_shape = tuple(s for s in out_shape if s != 1)
    if len(in_shape) == len(out_shape):
        return False
    shorter_shape = in_shape if len(in_shape) < len(out_shape) else out_shape
    longer_shape = in_shape if len(in_shape) > len(out_shape) else out_shape
    while len(longer_shape) > 0:
        if len(shorter_shape) == 0:
            return False
        elif longer_shape[0] == shorter_shape[0]:
            longer_shape = longer_shape[1:]
            shorter_shape = shorter_shape[1:]
        else:
            longer_shape = (longer_shape[0] * longer_shape[1],) + longer_shape[2:]
    if len(shorter_shape) > 0:
        return False
    return True

class State:
    def __init__(self, in_values, out_values, in_separator, out_separator, in_format, out_format, hints=[], verbose=False):
        if (in_format is None) != (out_format is None):
            raise ValueError("Either both or neither of in_format and out_format must be specified")
        if not in_format is None:
            if not in_format in format_map:
                raise ValueError(f"Format {in_format} not supported")
            if not out_format in format_map:
                raise ValueError(f"Format {out_format} not supported")
            self.in_format = format_map[in_format]
            self.out_format = format_map[out_format]
        else:
            self.in_format = self.out_format = None
        self.in_separator = in_separator
        self.out_separator = out_separator
        self.verbose = verbose
        self.hints = hints

        self.in_values = [types.SimpleNamespace(
                name=name,
                value=in_value,
                shape=tuple(in_value.shape),
                other=None,
            ) for name, in_value in in_values.items()]
        self.out_values = [types.SimpleNamespace(
                name=name,
                shape=tuple(out_value.shape),
                other=None,
            ) for name, out_value in out_values.items()]

        self.paired_nodes = []

    def pair_node(self, in_node, out_node):
        assert out_node.is_leaf() == in_node.is_leaf()
        assert out_node.other is None and in_node.other is None

        out_node.other = in_node
        in_node.other = out_node

        if out_node.is_leaf():
            out_node.value.other = in_node.value
            in_node.value.other = out_node.value

        self.paired_nodes.append((out_node, in_node))

    def get_format_permutation(self, out_shape, in_name, out_name):
        if self.in_format is not None and self.in_format != self.out_format:
            if self.in_format == "pytorch" and self.out_format == "tensorflow":
                if (in_name.endswith(".weight") or in_name == "weight") and len(out_shape) >= 2 or (in_name.endswith(".in_proj_weight") and len(out_shape) == 2):
                    return list(range(2, len(out_shape))) + [1, 0]
            elif self.in_format == "tensorflow" and self.out_format == "pytorch":
                print("weightbridge - warning: Tensorflow to PyTorch conversion is not tested") # TODO: test this
                if (out_name.endswith(".weight") or out_name == "weight") and len(out_shape) >= 2 or (out_name.endswith(".in_proj_weight") and len(out_shape) == 2):
                    return [len(out_shape) - 1, len(out_shape) - 2] + list(range(len(out_shape) - 2))
            else:
                raise ValueError(f"Conversion from {self.in_format} to {self.out_format} not supported")
        return None

    def adapt_format(self, value, out_shape, in_name, out_name):
        perm = self.get_format_permutation(out_shape, in_name, out_name)
        if not perm is None:
            perm_inv = [perm.index(i) for i in range(len(perm))]
            in_shape = [out_shape[i] for i in perm_inv]

            if value.shape != in_shape:
                assert can_reshape(value.shape, in_shape), f"Cannot reshape from {value.shape} to {in_shape} for {in_name} -> {out_name}"
                value = np.reshape(value, in_shape)
            value = np.transpose(value, perm)
            assert value.shape == out_shape, f"{value.shape} != {out_shape}"

        if value.shape != out_shape:
            assert can_reshape(value.shape, out_shape), f"Cannot reshape from {value.shape} to {out_shape} for {in_name} -> {out_name}"
            value = np.reshape(value, out_shape)

        return value

def matches_to_id(matches):
    return sorted(
        [(set(n.full_prefix for n in match.in_nodes), set(n.full_prefix for n in match.out_nodes)) for match in matches],
        key=lambda m: len(m[0]) + len(m[1]),
    )

def matcher(func):
    def wrapped(state, matches_in, *args, **kwargs):
        all_matches_out = []
        for match_in in matches_in:
            assert isinstance(match_in, Match)
            matches_out = list(func(state, match_in, *args, **kwargs))
            for match_out in matches_out:
                assert isinstance(match_out, Match)

            # All remaining nodes into single match
            match_remaining = match_in
            for match_out in matches_out:
                match_remaining = match_remaining - match_out
            matches_out.append(match_remaining)

            # Remove empty or unmatchable pairs
            i = 0
            while i < len(matches_out):
                if len(matches_out[i].in_nodes) == 0 or len(matches_out[i].out_nodes) == 0:
                    if not all(not n.is_leaf() for n in matches_out[i].in_nodes) or not all(not n.is_leaf() for n in matches_out[i].out_nodes):
                        def check(nodes, type):
                            if any(n.is_leaf() for n in nodes):
                                print(f"Matcher {func.__name__} yielded unmatched leafs:")
                                for n in nodes:
                                    if n.is_leaf():
                                        print(f"    {type} {n.full_prefix} {n.get_structured_shapes()}")
                        check(matches_out[i].in_nodes, "IN ")
                        check(matches_out[i].out_nodes, "OUT")
                        print()
                        print("Got OUT values:")
                        for v in state.out_values:
                            print(f"    OUT {v.name} {v.shape}")
                        print()
                        print("Got IN  values:")
                        for v in state.in_values:
                            print(f"    IN  {v.name} {v.shape}")
                        raise ValueError("Matcher yielded unmatched leafs")
                    del matches_out[i]
                else:
                    i += 1

            for match_out in matches_out:
                if len(match_out.in_nodes) == 1 and len(match_out.out_nodes) == 1:
                    state.pair_node(match_out.in_nodes[0], match_out.out_nodes[0])
                elif len(match_out.in_nodes) > 0 or len(match_out.out_nodes) > 0:
                    all_matches_out.append(match_out)
                else: # Empty match
                    pass

                if state.verbose and not match_out.description is None and (matches_to_id([match_in]) != matches_to_id(matches_out)):
                    print(match_out.description)
                    for n in match_out.out_nodes:
                        print(f"    OUT {n.full_prefix} {n.get_structured_shapes()}")
                    for n in match_out.in_nodes:
                        print(f"    IN  {n.full_prefix} {n.get_structured_shapes()}")

        changed = matches_to_id(matches_in) != matches_to_id(all_matches_out)
        return all_matches_out, changed
    wrapped.__name__ = func.__name__
    return wrapped

def hint_matcher(func):
    def wrapped(state, match):
        hints, is_match, hint_description = func(state, match)

        for hint in hints:
            # Build score matrix for all pairs of nodes
            matrix = np.zeros((len(match.out_nodes), len(match.in_nodes)), dtype="int32")
            for out_index in range(len(match.out_nodes)):
                for in_index in range(len(match.in_nodes)):
                    matrix[out_index, in_index] = 1 if is_match(match.out_nodes[out_index], match.in_nodes[in_index], hint) else 0
            s = np.sum(matrix)
            if s < len(match.out_nodes) or s == len(match.out_nodes) * len(match.in_nodes):
                continue

            # Find contiguous groups of matched nodes in matrix
            index_groups = set()
            checked_pairs = set()
            for out_index in range(len(match.out_nodes)):
                for in_index in range(len(match.in_nodes)):
                    if not (out_index, in_index) in checked_pairs:
                        if matrix[out_index, in_index] == 1:
                            out_indices = set()
                            in_indices = set()
                            out_indices.add(out_index)
                            in_indices.add(in_index)
                            for out_index2 in range(len(match.out_nodes)):
                                if matrix[out_index2, in_index] == 1:
                                    out_indices.add(out_index2)
                                    checked_pairs.add((out_index2, in_index))
                            for in_index2 in range(len(match.in_nodes)):
                                if matrix[out_index, in_index2] == 1:
                                    in_indices.add(in_index2)
                                    checked_pairs.add((out_index, in_index2))

                            out_indices = tuple(sorted(out_indices))
                            in_indices = tuple(sorted(in_indices))
                            index_groups.add((out_indices, in_indices))
                        checked_pairs.add((out_index, in_index))

            # Merge all groups with unequal number of nodes
            merged_out_indices = set()
            merged_in_indices = set()
            index_groups_out = set()
            for out_indices, in_indices in index_groups:
                if len(out_indices) == len(in_indices):
                    index_groups_out.add((out_indices, in_indices))
                else:
                    merged_out_indices.update(out_indices)
                    merged_in_indices.update(in_indices)
            if len(merged_out_indices) != len(merged_in_indices):
                continue
            if len(merged_out_indices) > 0:
                out_indices = tuple(sorted(merged_out_indices))
                in_indices = tuple(sorted(merged_in_indices))
                index_groups_out.add((out_indices, in_indices))
            index_groups = index_groups_out

            # End if more than one group was found
            if len(index_groups) > 1:
                remaining_out_nodes = list(match.out_nodes)
                remaining_in_nodes = list(match.in_nodes)
                for out_indices, in_indices in index_groups:
                    out_group = [match.out_nodes[i] for i in out_indices]
                    in_group = [match.in_nodes[i] for i in in_indices]
                    for n in out_group:
                        assert n in remaining_out_nodes
                        remaining_out_nodes.remove(n)
                    for n in in_group:
                        assert n in remaining_in_nodes
                        remaining_in_nodes.remove(n)
                    yield Match(in_group, out_group, f"Found subgroup resulting from {hint_description(hint)}")
                yield Match(remaining_in_nodes, remaining_out_nodes)
                return

        yield match
    wrapped.__name__ = func.__name__
    wrapped = matcher(wrapped)
    return wrapped