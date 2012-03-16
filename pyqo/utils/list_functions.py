'''
def _conjugate_indices(indices, ndim):
    """
    Return all numbers from 0 to ndim which are not in indices.
    """
    if isinstance(indices, int):
        indices = (indices,)
    return set(range(ndim)).difference(indices)
'''
def sorted_list(iterable, reverse=False):
    """
    Transform an iterable to a sorted list.
    """
    a = list(iterable)
    a.sort()
    if reverse:
        a.reverse()
    return a

