"""
Tools for initialization of the zero level set.
"""

class init_func_base(object):
    """
    The template class for initialization functions.

    At minimum, the `__call__` function should be implemented and
    return `u0, dist, mask`.
    """
    def __init__(self): pass
    def __call__(self, u, band):
        raise NotImplementedError
