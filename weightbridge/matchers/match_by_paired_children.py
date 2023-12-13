from .base import matcher, Match

@matcher
def match_by_paired_children(state, match):
    out_nodes = list(match.out_nodes)
    in_nodes = list(match.in_nodes)
    for pair_out_node, pair_in_node in state.paired_nodes:
        out_matches = [n for n in out_nodes if pair_out_node.full_prefix.startswith(n.full_prefix)]
        in_matches = [n for n in in_nodes if pair_in_node.full_prefix.startswith(n.full_prefix)]
        if len(out_matches) == 1 and len(in_matches) == 1:
            out_node = out_matches[0]
            in_node = in_matches[0]
            yield Match([in_node], [out_node], f"Matched by paired children of OUT {pair_out_node.full_prefix} IN {pair_in_node.full_prefix}")
            out_nodes.remove(out_node)
            in_nodes.remove(in_node)