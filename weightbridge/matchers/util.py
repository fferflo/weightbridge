import re

def extract_number_expression(name):
    expr = ""
    is_num = False
    nums = 0
    for c in name:
        if c.isnumeric():
            is_num = True
        else:
            if is_num:
                is_num = False
                expr += "([0-9]+)"
                nums += 1
            expr += re.escape(c)
    if is_num:
        expr += "([0-9]+)"
        nums += 1
    expr = "^" + expr + "$"
    return expr, nums