""" Tools for initialization of the zero level set.
"""
import abc

import numpy
import skfmm


class InitializationBase(abc.ABC):
    """ The template class for initialization functions.

    At minimum, the `__call__` function should be implemented and
    return `u0, dist, mask`.
    """

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, img, band, dx=None, seed=None):
        """
        All initialization functions must have the above __call__ signature
        with *at least* these arguments (additional keyword arguments are
        of course allowed, but won't be used by the `fit` routine
        in the `level_set_machine_learning` module).
        """

        # Validate the delta terms
        if dx is None:
            dx = numpy.ones(self.ndim, dtype=numpy.float)
        else:
            dx = numpy.array(dx, dtype=numpy.float)
            if len(dx) != self.ndim:
                msg = "Number of dx terms ({}) doesn't match dimensions ({})"
                raise ValueError(msg.format(len(dx), self.ndim))

        # Compute the initialization
        init_mask = self.initialize(img=img, dx=dx, seed=seed)

        # Validate the returned mask
        if not isinstance(init_mask, numpy.ndarray):
            msg = ("Returned initialization was type {} but "
                   "should be numpy.ndarray")
            raise TypeError(msg.format(type(init_mask)))

        if init_mask.dtype != numpy.bool:
            msg = "Returned initialization was dtype {} but should be bool"
            raise TypeError(msg.format(init_mask.dtype))

        if init_mask.shape != img.shape:
            msg = "Returned initialization was shape {} but should be {}"
            raise ValueError(msg.format(init_mask.shape, img.shape))

        # Set the initial level set function
        u = 2 * init_mask.astype(numpy.float) - 1

        # Initialize the distance transform
        dist = skfmm.distance(u, narrow=band, dx=dx)

        return u, dist, mask

    @abc.abstractmethod
    def initialize(self, img, dx=None, seed=None):
        raise NotImplementedError
