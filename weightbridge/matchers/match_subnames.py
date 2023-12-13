from .base import hint_matcher

@hint_matcher
def match_subnames(state, match):
    # Find subnames
    subnames = set()
    for separator, nodes in [(state.out_separator, match.out_nodes), (state.in_separator, match.in_nodes)]:
        for node in nodes:
            tokens = node.full_prefix.split(separator)
            tokens = tokens[:-1]
            subnames.update(tokens)

    def is_match(out_node, in_node, subname):
        return (subname in out_node.full_prefix) == (subname in in_node.full_prefix)

    return subnames, is_match, lambda hint: f"subname {hint}"