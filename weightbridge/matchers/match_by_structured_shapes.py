from collections import defaultdict
from .base import matcher, Match

def structured_shapes_depth(shape):
    if isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
        return 1
    else:
        assert all(not isinstance(s, int) for s in shape)
        return 1 + max(structured_shapes_depth(s) for s in shape)

@matcher
def match_by_structured_shapes(state, match, **kwargs):
    pairs_to_id = lambda nodes: [n.full_prefix for n in nodes]

    def to_uniquekey_dict(nodes, is_input):
        result = defaultdict(list)
        for n in nodes:
            if n.other is None:
                uniquekey = n.get_structured_shapes(is_input=is_input, state=state, **kwargs)
                if uniquekey is None:
                    return None
                result[uniquekey].append(n)
        return result
    out_uniquekey_to_nodes = to_uniquekey_dict(match.out_nodes, is_input=False)
    if out_uniquekey_to_nodes is None:
        return
    in_uniquekey_to_nodes = to_uniquekey_dict(match.in_nodes, is_input=True)
    if in_uniquekey_to_nodes is None:
        return

    intersection_keys = set(out_uniquekey_to_nodes.keys()).intersection(in_uniquekey_to_nodes.keys())
    intersection_keys = sorted(intersection_keys, key=lambda shapes: -structured_shapes_depth(shapes))

    for key in intersection_keys:
        yield Match(in_uniquekey_to_nodes[key], out_uniquekey_to_nodes[key], f"Found matches with structured shapes {key}")

    for key in out_uniquekey_to_nodes.keys():
        if key not in intersection_keys:
            yield Match([], out_uniquekey_to_nodes[key])
    for key in in_uniquekey_to_nodes.keys():
        if key not in intersection_keys:
            yield Match(in_uniquekey_to_nodes[key], [])