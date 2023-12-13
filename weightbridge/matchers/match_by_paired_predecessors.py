from .base import hint_matcher

@hint_matcher
def match_by_paired_predecessors(state, match):
    def get_predecessors(node, nodes):
        parents = [node] if any(node.is_predecessor_of(n) for n in nodes) else []
        if not node.parent is None:
            parents = get_predecessors(node.parent, nodes) + parents
        return parents

    hints = set()
    for out_node, in_node in state.paired_nodes:
        out_predecessors = get_predecessors(out_node, match.out_nodes)
        in_predecessors = get_predecessors(in_node, match.in_nodes)
        for out_predecessor in out_predecessors:
            for in_predecessor in in_predecessors:
                hints.add((out_predecessor, in_predecessor))
    hints = sorted(hints, key=lambda t: -(t[0].depth + t[1].depth))

    def is_match(out_node, in_node, hint):
        return hint[0].is_predecessor_of(out_node) == hint[1].is_predecessor_of(in_node)

    return hints, is_match, lambda hint: f"paired predecessor OUT {hint[0].full_prefix} IN {hint[1].full_prefix}"