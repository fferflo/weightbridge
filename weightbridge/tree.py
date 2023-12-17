import functools
import numpy as np

def get_common_prefix(strings):
    strings = list(strings)
    s = ""
    while all(len(s) < len(sn) for sn in strings) and len(set(sn[len(s)] for sn in strings)) == 1:
        s = s + strings[0][len(s)]
    return s

class Node:
    def __init__(self, value, relative_prefix, full_prefix, direct_children, depth):
        self.value = value
        self.relative_prefix = relative_prefix
        self.full_prefix = full_prefix
        self.direct_children = direct_children
        self.parent = None
        self.depth = depth
        self.other = None
        if self.is_leaf():
            assert not value is None

    def is_leaf(self):
        return len(self.direct_children) == 0

    def is_predecessor_of(self, other):
        return id(other) == id(self) or (not other.parent is None and self.is_predecessor_of(other.parent))

    def get_structured_shapes(self, format=None, ignore_shape_one=True, remove_trivial_nodes=False, flatten=False, is_input=None, state=None):
        if self.value is None:
            value_shape = []
        else:
            if format is None:
                value_shape = self.value.shape
            elif format == "as-input-shape":
                if is_input:
                    value_shape = self.value.shape
                else:
                    value_shape = state.formatter.outshape_to_inshape(self.value.shape, self.full_prefix)
            elif format == "as-output-shape":
                if is_input:
                    value_shape = state.formatter.inshape_to_outshape(self.value.shape, self.full_prefix)
                else:
                    value_shape = self.value.shape
            elif format == "product":
                value_shape = (int(np.prod(self.value.shape)),)

            if value_shape is None:
                return None

            if ignore_shape_one:
                value_shape = tuple(s for s in value_shape if s != 1)

            value_shape = [value_shape]


        direct_children = self.direct_children
        if remove_trivial_nodes:
            def drop(n):
                if len(n.direct_children) == 1:
                    return drop(n.direct_children[0])
                else:
                    return n
            direct_children = [drop(c) for c in direct_children]
        children_shapes = [c.get_structured_shapes(
            format=format,
            ignore_shape_one=ignore_shape_one,
            remove_trivial_nodes=remove_trivial_nodes,
            flatten=flatten,
            is_input=is_input,
            state=state,
        ) for c in direct_children]
        if any(s is None for s in children_shapes):
            return None
        if flatten:
            def flatten(shapes):
                result = []
                for s in shapes:
                    if isinstance(s, tuple) and all(not isinstance(x, tuple) for x in s):
                        result.append(s)
                    else:
                        result.extend(flatten(s))
                return result
            children_shapes = flatten(children_shapes)
        def compare_int(i1, i2):
            if i1 < i2:
                return -1
            elif i1 > i2:
                return 1
            else:
                return 0
        def compare(shapes1, shapes2):
            if isinstance(shapes1, tuple) and isinstance(shapes2, tuple):
                c = compare_int(len(shapes1), len(shapes2))
                if c != 0:
                    return c
                for s1, s2 in zip(shapes1, shapes2):
                    c = compare(s1, s2)
                    if c != 0:
                        return c
                return 0
            elif isinstance(shapes1, int) and isinstance(shapes2, tuple):
                return 1
            elif isinstance(shapes1, tuple) and isinstance(shapes2, int):
                return -1
            else:
                assert isinstance(shapes1, int) and isinstance(shapes2, int)
                return compare_int(shapes1, shapes2)
        return tuple(sorted(value_shape + children_shapes, key=functools.cmp_to_key(compare)))

    def __iter__(self):
        yield self
        for c in self.direct_children:
            yield from c

    def printout(self, indentation=""):
        print(indentation, self.relative_prefix)
        if not self.value is None:
            print(indentation + "    " + self.value.name + " " + str(self.value.shape))
        for c in self.direct_children:
            c.printout(indentation + "    ")

def build(values, value_names_without_prefix=None, parent_prefix="", depth=0):
    if value_names_without_prefix is None:
        value_names_without_prefix = [v.name for v in values]
    assert len(values) == len(value_names_without_prefix)
    assert len(values) > 0
    values = list(values)
    value_names_without_prefix = list(value_names_without_prefix)

    prefix = get_common_prefix(value_names_without_prefix)

    node_value = None
    children = {}
    for value, value_name_without_prefix in zip(values, value_names_without_prefix):
        if len(value_name_without_prefix) == len(prefix):
            assert node_value is None
            node_value = value
        else:
            c = value_name_without_prefix[len(prefix)]
            if not c in children:
                children[c] = ([], [])
            children[c][0].append(value)
            children[c][1].append(value_name_without_prefix[len(prefix):])
    children = [build(values, value_names_without_prefix, parent_prefix=parent_prefix + prefix, depth=depth + 1) for values, value_names_without_prefix in children.values()]

    result = Node(
        value=node_value,
        relative_prefix=prefix,
        full_prefix=parent_prefix + prefix,
        direct_children=children,
        depth=depth,
    )
    for n in children:
        n.parent = result

    return result