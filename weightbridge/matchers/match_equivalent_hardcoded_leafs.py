from .base import hint_matcher

@hint_matcher
def match_equivalent_hardcoded_leafs(state, match):
    hints = [
        [
            ["weight", "scale", "gamma", "w"],
            ["bias", "offset", "beta", "b"],
            ["moving_mean", "running_mean", "mean_ema/average", "mean/average", "mean"],
            ["moving_variance", "moving_var", "running_variance", "running_var", "var_ema/average", "var/average", "var", "variance"],
        ]
    ]

    out_nodes = [n for n in match.out_nodes if n.is_leaf()]
    in_nodes = [n for n in match.in_nodes if n.is_leaf()]

    def get_index(node, separator, hint):
        for i, equivalent_postfixes in enumerate(hint):
            if any((node.full_prefix.endswith(separator + postfix) or node.full_prefix == postfix) for postfix in equivalent_postfixes):
                return i
        return -1

    def is_valid_hint(hint):
        return all(get_index(n, state.out_separator, hint) >= 0 for n in out_nodes) and all(get_index(n, state.in_separator, hint) >= 0 for n in in_nodes)

    hints = [h for h in hints if is_valid_hint(h)]

    def is_match(out_node, in_node, hint):
        return get_index(out_node, state.out_separator, hint) == get_index(in_node, state.in_separator, hint)

    return hints, is_match, lambda hint: f"equivalent hardcoded leafs"