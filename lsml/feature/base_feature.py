""" Abstract base classes for features
"""
import abc
import numpy


IMAGE_FEATURE_TYPE = 'image'
SHAPE_FEATURE_TYPE = 'shape'
LOCAL_FEATURE_TYPE = 'local'
GLOBAL_FEATURE_TYPE = 'global'


class BaseFeature(abc.ABC):
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
        return self.name

    def __repr__(self):
        return self.__str__()

    def __init__(self, ndim=2):
        """ Initialize the feature

        Parameters
        ----------
        ndim: int
            The dimensions of the input space

        """
        self.ndim = ndim

    def __call__(self, u, img=None, dist=None, mask=None, dx=None):
        """ Calls the feature computation function after performing
        some validation on the inputs
        """

        # Error message for incorrect variable type
        not_ndarray = "`{var}` must be a numpy array"

        if not isinstance(u, numpy.ndarray):
            raise TypeError(not_ndarray.format(var='u'))

        # Set the shape
        shape = u.shape

        # Error message for shape mismatches
        shape_mismatch = ("`{{var}}` has shape {{shape}}, "
                          "but must be shape {correct_shape}")
        shape_mismatch = shape_mismatch.format(correct_shape=shape)

        if isinstance(self, BaseImageFeature):
            if img is None:
                msg = "`img` must be present for image features"
                raise ValueError(msg)
            if not isinstance(img, numpy.ndarray):
                raise TypeError(not_ndarray.format(var='img'))
            if img.shape != shape:
                msg = shape_mismatch.format(var='img', shape=img.shape)
                raise ValueError(msg)

        if dist is not None:
            if not isinstance(dist, numpy.ndarray):
                raise TypeError(not_ndarray.format(var='dist'))
            if dist.shape != shape:
                msg = shape_mismatch.format(var='img', shape=dist.shape)
                raise ValueError(msg)

        # Check delta terms
        if dx is None:
            dx = numpy.ones(self.ndim, dtype=numpy.float)
        else:
            dx = numpy.array(dx, dtype=numpy.float)
            if len(dx) != self.ndim:
                msg = "Number of dx terms ({}) doesn't match dimensions ({})"
                raise ValueError(msg.format(len(dx), self.ndim))

        if mask is None:
            mask = numpy.ones(shape, dtype=numpy.bool)

        # Check mask is a numpy array
        if not isinstance(mask, numpy.ndarray):
            msg = not_ndarray.format(var='mask')
            raise TypeError(msg)

        # Check shape of mask
        if mask.shape != shape:
            msg = shape_mismatch.format(var='mask', shape=mask.shape)
            raise ValueError(msg)

        # Check dtype of mask
        if mask.dtype != numpy.bool:
            msg = "`mask` dtype ({}) was not of type bool"
            raise TypeError(msg.format(mask.dtype))

        # Handle the empty mask case
        if not mask.any():
            return numpy.empty_like(u)

        if isinstance(self, BaseImageFeature):
            return self.compute_feature(
                u=u, img=img, dist=dist, mask=mask, dx=dx)
        elif isinstance(self, BaseShapeFeature):
            return self.compute_feature(
                u=u, dist=dist, mask=mask, dx=dx)

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
    type = IMAGE_FEATURE_TYPE

    def __init__(self, ndim=2, sigma=3):
        """ Initialize a smoothed image standard deviation feature

        ndim: int
            The number of dimensions in which the feature will be computed

        sigma: float
            The smooth parameter for Gaussian smoothing (note that
            sigma = 0 yields no smoothing; also note that anisotropic
            volumes will alter sigma along each axis according to the
            provided dx terms)

        """
        super(BaseImageFeature, self).__init__(ndim)
        self.sigma = sigma


class BaseShapeFeature(BaseFeature):
    """ The abstract base class for all shape features
    """
    type = SHAPE_FEATURE_TYPE
