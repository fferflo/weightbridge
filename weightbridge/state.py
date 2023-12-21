from . import format
import types

class State:
    def __init__(self, in_values, out_values, in_separator, out_separator, in_format, out_format, hints=[], ignore_unmatched_inputs=False, ignore_unmatched_outputs=False, verbose=False):
        self.formatter = format.Formatter(in_format, out_format, in_separator, out_separator)
        self.in_separator = in_separator
        self.out_separator = out_separator
        self.verbose = verbose
        self.hints = hints
        self.ignore_unmatched_inputs = ignore_unmatched_inputs
        self.ignore_unmatched_outputs = ignore_unmatched_outputs
        self.used_hints = set()

        self.in_values = [types.SimpleNamespace(
                name=name,
                value=in_value,
                shape=tuple(in_value.shape),
                other=None,
            ) for name, in_value in in_values.items()]
        self.out_values = [types.SimpleNamespace(
                name=name,
                value=out_value,
                shape=tuple(out_value.shape),
                other=None,
            ) for name, out_value in out_values.items()]

        self.paired_nodes = []

    def pair_node(self, in_node, out_node):
        assert out_node.is_leaf() == in_node.is_leaf()
        assert out_node.other is None and in_node.other is None

        out_node.other = in_node
        in_node.other = out_node

        if out_node.is_leaf():
            out_node.value.other = in_node.value
            in_node.value.other = out_node.value

        self.paired_nodes.append((out_node, in_node))

        self.formatter.pair(in_node.full_prefix, out_node.full_prefix)