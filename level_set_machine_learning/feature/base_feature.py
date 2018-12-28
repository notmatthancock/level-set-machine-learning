""" Abstract base classes for features
"""
import abc
import numpy


IMAGE_FEATURE_TYPE = 'image'
SHAPE_FEATURE_TYPE = 'shape'
LOCAL_FEATURE_TYPE = 'local'
GLOBAL_FEATURE_TYPE = 'global'


class BaseFeature(object):
    """ The abstract base class for all features
    """

    @property
    @abc.abstractmethod
    def name(self):
        """ The name of the feature
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def type(self):
        """ The type of the feature, e.g., image or shape
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def locality(self):
        """ The locality of the feature, e.g., local versus global
        """
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "<{} {}>".format(self.__class__.__name__,
                                self.name)

    def __init__(self, ndim):
        """ Initialize the feature

        Parameters
        ----------
        ndim: int
            The dimensions of the input space

        """
        self.ndim = ndim

    def __call__(self, u, img, dist, mask, dx=None):
        """ Calls the feature computation function after performing
        some validation on the inputs
        """
        # Check shapes
        shape = u.shape
        if img.shape != shape or dist.shape != shape or mask.shape != shape:
            shapes = (u.shape, img.shape, dist.shape, mask.shape)
            msg = ("Shape mismatch in one of the inputs: u={}, img={}, "
                   "dist={}, mask={}")
            raise ValueError(msg.format(*shapes))

        # Check dimensions
        ndim = self.ndim
        if img.ndim != ndim or dist.ndim != ndim or mask.ndim != ndim:
            ndims = (u.ndim, img.ndim, dist.ndim, mask.ndim)
            msg = ("Shape mismatch in one of the inputs: u={}, img={}, "
                   "dist={}, mask={}")
            raise ValueError(msg.format(*ndims))

        # Check delta terms
        if dx is None:
            dx = numpy.ones(self.ndim)
        else:
            dx = numpy.array(dx)
            if len(dx) != self.ndim:
                msg = "Number of dx terms ({}) doesn't match dimensions ({})"
                raise ValueError(msg.format(len(dx), self.ndim))

        # Check dtype of mask
        if mask.dtype != numpy.bool:
            msg = "mask dtype ({}) was not of type bool"
            raise ValueError(msg.format(mask.dtype))

        return self.compute_feature(u=u, img=img, dist=dist, mask=mask, dx=dx)

    @abc.abstractmethod
    def compute_feature(self, u, img, dist, mask, dx):
        """ Compute the feature

        Parameters
        ----------
        u: numpy.array
            The "level set function".

        img: numpy.array
            The image/image volume/image hyper-volume.

        dist: numpy.array
            The signed distance transform to the level set `u` (only computed
            necessarily in the narrow band region).

        mask: numpy.array (dtype=bool)
            The boolean mask indicating the narrow band region of `u`.

        dx: numpy.array, shape=ndim
            The "delta" spacing terms for each image axis. If None, then
            1.0 is used for each axis. This array should be in index space,
            i.e., the first term should refer to spacing along the first
            axis, etc.

        Returns
        -------
        feature: numpy.array, shape=u.shape

        """
        raise NotImplementedError


class BaseImageFeature(BaseFeature):
    """ The abstract base class for all image features
    """
    type = 'image'


class BaseShapeFeature(BaseFeature):
    """ The abstract base class for all shape features
    """
    type = 'shape'

    def __call__(self, u, dist, mask, dx=None):
        """ Calls the feature computation function after performing
        some validation on the inputs
        """
        # Check ndims
        ndim = self.ndim
        if dist.ndim != ndim or mask.ndim != ndim:
            ndims = (ndim, u.ndim, dist.ndim, mask.ndim)
            msg = ("One of the inputs is not the correct dimension "
                   "(correct={}): u={}, dist={}, mask={}")
            raise ValueError(msg.format(*ndims))

        # Check shapes
        shape = u.shape
        if dist.shape != shape or mask.shape != shape:
            shapes = (u.shape, dist.shape, mask.shape)
            msg = ("Shape mismatch in one of the inputs: u={}, "
                   "dist={}, mask={}")
            raise ValueError(msg.format(*shapes))

        # Check delta terms
        if dx is None:
            dx = numpy.ones(self.ndim)
        else:
            dx = numpy.array(dx)
            if len(dx) != self.ndim:
                msg = "Number of dx terms ({}) doesn't match dimensions ({})"
                raise ValueError(msg.format(len(dx), self.ndim))

        # Check dtype of mask
        if mask.dtype != numpy.bool:
            msg = "mask dtype ({}) was not of type bool"
            raise ValueError(msg.format(mask.dtype))

        return self.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

    @abc.abstractmethod
    def compute_feature(self, u, dist, mask, dx):
        """ Compute the feature

        Parameters
        ----------
        u: numpy.array
            The "level set function".

        dist: numpy.array
            The signed distance transform to the level set `u` (only computed
            necessarily in the narrow band region).

        mask: numpy.array (dtype=bool)
            The boolean mask indicating the narrow band region of `u`.

        dx: numpy.array, shape=ndim
            The "delta" spacing terms for each image axis. If None, then
            1.0 is used for each axis.

        Returns
        -------
        feature: numpy.array, shape=u.shape
        """
        raise NotImplementedError
