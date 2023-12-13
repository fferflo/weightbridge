from .base import matcher, Match
from .util import extract_number_expression
import re

@matcher
def match_known_paired_number_regex(state, match):
    last_pairs_num_name = "match_known_paired_number_regex_last_pairs_num"
    if not last_pairs_num_name in vars(state):
        vars(state)[last_pairs_num_name] = 0
    hints_name = "match_known_paired_number_regex_hints"
    if not hints_name in vars(state):
        vars(state)[hints_name] = {}
    hints = vars(state)[hints_name]

    for out_node, in_node in state.paired_nodes[vars(state)[last_pairs_num_name]:]:
        out_expr, out_nums = extract_number_expression(out_node.full_prefix)
        in_expr, in_nums = extract_number_expression(in_node.full_prefix)
        if out_nums > 0 and in_nums > 0:
            hint = (out_expr, in_expr)
            if not hint in hints:
                hints[hint] = (re.compile(hint[0]), re.compile(hint[1]))
    vars(state)[last_pairs_num_name] = len(state.paired_nodes)

    def is_match(name, expr):
        re_match = re.match(expr, name)
        if re_match:
            return tuple(int(g) for g in re_match.groups())
        else:
            return None

    out_nodes = list(match.out_nodes)
    in_nodes = list(match.in_nodes)
    for hint in hints.values():
        def get_matches(nodes, hint):
            matches = [(n, is_match(n.full_prefix, hint)) for n in nodes]
            matches = [(n, nums) for n, nums in matches if not nums is None]
            matches = sorted(matches, key=lambda t: t[1])
            nums = [num for _, nums in matches for num in nums]
            if len(nums) != len(set(nums)):
                return None
            matches = [n for n, nums in matches]
            return matches
        out_matching_nodes = get_matches(out_nodes, hint[0])
        in_matching_nodes = get_matches(in_nodes, hint[1])
        if not out_matching_nodes is None and not in_matching_nodes is None and len(out_matching_nodes) == len(in_matching_nodes) and len(out_matching_nodes) > 0:
            for out_node, in_node in zip(out_matching_nodes, in_matching_nodes):
                yield Match([in_node], [out_node], f"Found counted nodes from paired hint OUT {hint[0].pattern[1:-1]} IN  {hint[1].pattern[1:-1]}:")
                out_nodes.remove(out_node)
                in_nodes.remove(in_node)
