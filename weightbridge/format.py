import einx
import numpy as np

leaf_types = {
    "weight": "weight",
    "kernel": "weight",
    "w": "weight",
}

class Format:
    def __init__(self, expressions):
        self.expressions = expressions

formats = {
}

formats["pytorch"] = formats["jittor"] = Format({"weight": "o i k..."})
formats["flax"] = formats["haiku"] = formats["tensorflow"] = formats["tf"] = Format({"weight": "k... i o"})



def can_reshape(in_shape, out_shape):
    if tuple(s for s in in_shape if s != 1) == tuple(s for s in out_shape if s != 1):
        return True
    in_shape = tuple(s for s in in_shape if s != 1)
    out_shape = tuple(s for s in out_shape if s != 1)
    if len(in_shape) == len(out_shape):
        return False
    shorter_shape = in_shape if len(in_shape) < len(out_shape) else out_shape
    longer_shape = in_shape if len(in_shape) > len(out_shape) else out_shape
    while len(longer_shape) > 0:
        if len(shorter_shape) == 0:
            return False
        elif longer_shape[0] == shorter_shape[0]:
            longer_shape = longer_shape[1:]
            shorter_shape = shorter_shape[1:]
        else:
            longer_shape = (longer_shape[0] * longer_shape[1],) + longer_shape[2:]
    if len(shorter_shape) > 0:
        return False
    return True

def _input_format(format):
    if format is None:
        return None
    elif isinstance(format, str):
        if not format in formats:
            raise ValueError(f"Unknown format {format}")
        return formats[format]
    else:
        if not isinstance(format, Format):
            raise ValueError(f"Unknown format {format}")
        return format

class Formatter:
    def __init__(self, in_format, out_format, in_separator, out_separator):
        if (in_format is None) != (out_format is None):
            raise ValueError("Either both or neither of in_format and out_format must be specified")
        self.in_format = _input_format(in_format)
        self.out_format = _input_format(out_format)
        self.in_separator = in_separator
        self.out_separator = out_separator
        self.in_leaf_types = dict(leaf_types)
        self.out_leaf_types = dict(leaf_types)

    def pair(self, in_name, out_name):
        in_leaf = in_name.split(self.in_separator)[-1]
        out_leaf = out_name.split(self.out_separator)[-1]

        in_leaf_type = self.in_leaf_types.get(in_leaf, None)
        out_leaf_type = self.out_leaf_types.get(out_leaf, None)
        if not in_leaf_type is None and not out_leaf_type is None and in_leaf_type != out_leaf_type:
            raise ValueError(f"Got incompatible leafs {in_leaf} and {out_leaf}") # TODO: exception type
        leaf_type = in_leaf_type if not in_leaf_type is None else out_leaf_type
        self.in_leaf_types[in_leaf] = leaf_type
        self.out_leaf_types[out_leaf] = leaf_type

        return leaf_type

    def outshape_to_inshape(self, out_shape, out_name):
        if self.out_format is None or len(out_shape) <= 1:
            return out_shape

        leaf_type = self.out_leaf_types.get(out_name.split(self.out_separator)[-1], None)
        if leaf_type is None:
            return None

        expr_in = self.in_format.expressions[leaf_type]
        expr_out = self.out_format.expressions[leaf_type]

        (expr_out,), (expr_in,) = einx.rearrange.parse(f"{expr_out} -> {expr_in}", tuple(out_shape))
        return expr_in.shape

    def inshape_to_outshape(self, in_shape, in_name):
        if self.in_format is None or len(in_shape) <= 1:
            return in_shape

        leaf_type = self.in_leaf_types.get(in_name.split(self.in_separator)[-1], None)
        if leaf_type is None:
            return None

        expr_in = self.in_format.expressions[leaf_type]
        expr_out = self.out_format.expressions[leaf_type]

        (expr_in,), (expr_out,) = einx.rearrange.parse(f"{expr_in} -> {expr_out}", tuple(in_shape))
        return expr_out.shape

    def adapt_format(self, value, out_shape, in_name, out_name):
        if not self.in_format is None and len(out_shape) > 1:
            leaf_type = self.pair(in_name, out_name)

            if not leaf_type is None:
                expr_in = self.in_format.expressions[leaf_type]
                expr_out = self.out_format.expressions[leaf_type]

                (expr_out,), (expr_in,) = einx.rearrange.parse(f"{expr_out} -> {expr_in}", tuple(out_shape))
                in_shape = expr_in.shape

                if value.shape != in_shape:
                    assert can_reshape(value.shape, in_shape), f"Cannot reshape from {value.shape} to {in_shape} for {in_name} -> {out_name}"
                    value = np.reshape(value, in_shape)

                (value,), _ = einx.rearrange_stage3([expr_in], [value], [expr_out])

        if value.shape != out_shape:
            assert can_reshape(value.shape, out_shape), f"Cannot reshape from {value.shape} to {out_shape} for {in_name} -> {out_name}"
            value = np.reshape(value, out_shape)

        return value