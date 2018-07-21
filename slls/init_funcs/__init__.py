"""
Tools for initialization of the zero level set.
"""

class init_func_base(object):
    """
    The template class for initialization functions.

    At minimum, the `__call__` function should be implemented and
    return `u0, dist, mask`.
    """
    def __init__(self):
        raise NotImplementedError
    def __call__(self, img, band, dx=None, seed=None):
        """
        All initialization functions must have the above __call__ signature
        with *at least* these arguments (additional keyword arguments are
        of course allowed, but won't be used by the `fit` routine
        in the `slls` module).
        """
        raise NotImplementedError
