from .base import hint_matcher

@hint_matcher
def match_passed_hints(state, match):
    def is_match(out_node, in_node, hint):
        return (hint[0] in out_node.full_prefix) == (hint[1] in in_node.full_prefix)

    return state.hints, is_match, lambda hint: f"passed hints"