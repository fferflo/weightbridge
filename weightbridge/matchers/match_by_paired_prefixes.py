from .base import hint_matcher

@hint_matcher
def match_by_paired_prefixes(state, match):
    def get_prefixes(name, separator, nodes):
        prefixes = []
        prefix = ""
        for t in name.split(separator):
            prefix = (prefix + separator + t) if len(prefix) > 0 else t
            prefixes.append(prefix)
        def check(prefix):
            starting_with = list(n.full_prefix.startswith(prefix) for n in nodes)
            return any(starting_with)
        prefixes = [p for p in prefixes if check(p)]
        return prefixes

    hints = set()
    for out_node, in_node in state.paired_nodes:
        out_prefixes = get_prefixes(out_node.full_prefix, state.out_separator, match.out_nodes)
        in_prefixes = get_prefixes(in_node.full_prefix, state.in_separator, match.in_nodes)
        for out_prefix in out_prefixes:
            for in_prefix in in_prefixes:
                hints.add((out_prefix, in_prefix))
    hints = sorted(hints, key=lambda t: -(len(t[0]) + len(t[1])))

    def is_match(out_node, in_node, hint):
        return out_node.full_prefix.startswith(hint[0]) == in_node.full_prefix.startswith(hint[1])

    return hints, is_match, lambda hint: f"paired prefix hint OUT {hint[0]} IN  {hint[1]}"