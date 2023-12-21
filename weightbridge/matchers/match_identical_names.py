from .base import matcher, Match

@matcher
def match_identical_names(state, match):
    in_nodes = {node.full_prefix: node for node in match.in_nodes}
    out_nodes = {node.full_prefix: node for node in match.out_nodes}
    if in_nodes.keys() == out_nodes.keys():
        for name in in_nodes.keys():
            yield Match([in_nodes[name]], [out_nodes[name]], f"Found identical names {name}")
