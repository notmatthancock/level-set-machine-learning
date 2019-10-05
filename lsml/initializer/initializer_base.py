import abc

import numpy

from lsml.util.distance_transform import (
    distance_transform)


class InitializerBase(abc.ABC):
    """ The abstract base class for level set initializer functions.
    """

    def __init__(self):
        """ Supply the initialization instance with attributes that are
        necessary for re-use (e.g., a threshold parameter or a random state)
        """
        pass

    def __call__(self, img, band=0, dx=None, seed=None):
        """ The __call__ function handles input validation, etc. This
        function is used internally and calls the user-implemented
        `initializer` member function.
        """
        # Validate the delta terms
        if dx is None:
            dx = numpy.ones(img.ndim, dtype=numpy.float)
        else:
            dx = numpy.array(dx, dtype=numpy.float)
            if len(dx) != img.ndim:
                msg = "Number of dx terms ({}) doesn't match dimensions ({})"
                raise ValueError(msg.format(len(dx), img.ndim))

        if seed is None:
            seed = numpy.array(img.shape) / 2
        else:
            seed = numpy.array(seed)

        # Compute the initializer
        init_mask = self.initialize(img=img, dx=dx, seed=seed)

        # Validate the returned mask
        if not isinstance(init_mask, numpy.ndarray):
            msg = ("Returned initializer was type {} but "
                   "should be numpy.ndarray")
            raise TypeError(msg.format(type(init_mask)))

        if init_mask.dtype != numpy.bool:
            msg = "Returned initializer was dtype {} but should be bool"
            raise TypeError(msg.format(init_mask.dtype))

        if init_mask.shape != img.shape:
            msg = "Returned initializer was shape {} but should be {}"
            raise ValueError(msg.format(init_mask.shape, img.shape))

        # Set the initial level set function
        u = 2 * init_mask.astype(numpy.float) - 1

        # Compute the distance transform
        dist, mask = distance_transform(arr=u, band=band, dx=dx)

        return u, dist, mask

    @abc.abstractmethod
    def initialize(self, img, dx, seed):
        raise NotImplementedError
