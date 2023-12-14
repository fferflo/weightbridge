from .base import matcher, Match

@matcher
def match_by_paired_parents(state, match):
    def get_first_paired_parent(node):
        if not node.other is None:
            return node
        elif node.parent is None:
            return None
        else:
            return get_first_paired_parent(node.parent)

    def to_dict(nodes):
        pairedparent_to_nodes = {}
        for n in nodes:
            parent = get_first_paired_parent(n)
            if not parent is None:
                if not parent.full_prefix in pairedparent_to_nodes:
                    pairedparent_to_nodes[parent.full_prefix] = (parent, [])
                pairedparent_to_nodes[parent.full_prefix][1].append(n)
        return pairedparent_to_nodes

    # Link every paired parent to nodes in the current match-group
    out_pairedparent_to_nodes = to_dict(match.out_nodes)
    in_pairedparent_to_nodes = to_dict(match.in_nodes)

    for out_parent, out_children in out_pairedparent_to_nodes.values():
        in_parent = out_parent.other
        in_children = in_pairedparent_to_nodes[in_parent.full_prefix][1]
        yield Match(in_children, out_children, f"Found subtrees with matched parents OUT {out_parent.full_prefix} IN  {in_parent.full_prefix}")