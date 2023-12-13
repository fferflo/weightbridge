from collections import defaultdict
import numpy as np
from .base import matcher, Match
from .util import extract_number_expression
import re

@matcher
def match_number_regex(state, match):
    def make_groups(nodes):
        nodes = [n for n in nodes]
        groupswithmatchingregex = []
        while len(nodes) > 1:
            # Find possible prefixes
            prefix_expressions = set()
            for expr, _ in set(extract_number_expression(n.full_prefix) for n in nodes):
                prefix = ""
                for t in expr.split("([0-9]+)")[:-1]:
                    prefix = prefix + t + "([0-9]+)"
                    prefix_expressions.add(prefix)
            prefix_expressions = sorted(prefix_expressions, key=lambda expr: len(expr))

            for prefix_expr in prefix_expressions:
                prefix_expr = re.compile(prefix_expr)
                matches = [prefix_expr.match(n.full_prefix) for n in nodes]
                if sum((1 if m else 0) for m in matches) > 1:
                    matching_nodes = [n for n, m in zip(nodes, matches) if m]
                    matches = [m for m in matches if m]

                    # Find the first varying number in the list of numbers per node
                    numbers = np.asarray([[int(g) for g in match.groups()] for match in matches]) # matches nums
                    unique_nums = np.asarray([len(set(numbers[:, num_index])) for num_index in range(numbers.shape[1])]) # nums
                    for used_num_index in range(len(unique_nums)):
                        if unique_nums[used_num_index] > 1:
                            break
                    else:
                        continue
                    numbers = numbers[:, used_num_index] # matches

                    groupswithmatchingregex.append([(n, m) for n, m in zip(numbers, matching_nodes)])
                    for n in matching_nodes:
                        nodes.remove(n)
                    break # Found a prefix that counts more than one node
            else:
                break # No prefix matches

        for n in nodes:
            groupswithmatchingregex.append([(None, n)])

        num_to_groupswithmatchingregex = defaultdict(list)
        for g in groupswithmatchingregex:
            num_to_groupswithmatchingregex[len(g)].append(g)

        return num_to_groupswithmatchingregex

    # Divide nodes into groups that look like: prefix number postfix
    out_num_to_groupswithmatchingregex = make_groups(match.out_nodes)
    in_num_to_groupswithmatchingregex = make_groups(match.in_nodes)

    # Pair groups that have the same unique size
    unique_nums = set(out_num_to_groupswithmatchingregex.keys()).intersection(set(in_num_to_groupswithmatchingregex.keys()))
    for k in unique_nums:
        out_groups = out_num_to_groupswithmatchingregex[k]
        in_groups = in_num_to_groupswithmatchingregex[k]
        if len(out_groups) == 1 and len(in_groups) == 1:
            # Unique number of nodes -> divide by parsed number values
            def to_dict(group):
                nums_to_nodes = defaultdict(list)
                for num, node in group:
                    nums_to_nodes[num].append(node)
                return nums_to_nodes
            out_nums_to_nodes = to_dict(out_groups[0])
            in_nums_to_nodes = to_dict(in_groups[0])
            if len(out_nums_to_nodes) == len(in_nums_to_nodes):
                # Sort by parsed number values
                out_sorted_groups = [nodes for num, nodes in sorted(out_nums_to_nodes.items(), key=lambda t: t[0])]
                in_sorted_groups = [nodes for num, nodes in sorted(in_nums_to_nodes.items(), key=lambda t: t[0])]
                assert len(out_sorted_groups) == len(in_sorted_groups)

                # Match
                for out_group, in_group in zip(out_sorted_groups, in_sorted_groups):
                    yield Match(in_group, out_group, "Found counted modules")