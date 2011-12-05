import types

def _conjugate_indices(indices, ndim):
    """
    Return all numbers from 0 to ndim which are not in indices.
    """
    if isinstance(indices, int):
        indices = (indices,)
    return set(range(ndim)).difference(indices)

def _sorted_list(iterable, reverse=False):
    """
    Transform an iterable to a sorted list.
    """
    a = list(iterable)
    a.sort()
    if reverse:
        a.reverse()
    return a

def add_methods(array, methods):
    for name, func in methods.items():
        f = types.MethodType(func, array)
        setattr(array, name, f)

def add_properties(array, properties):
    for name, func in properties.items():
        f = types.MethodType(func, array)
        setattr(array, "_"+name, f)

