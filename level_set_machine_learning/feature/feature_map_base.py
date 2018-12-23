"""
Abstract base class for feature map functions.

Use this class is a "template" for creating feature map classes.

For a specific example, see the `dim2/simple_feature_map.py` class.
"""
import abc

import numpy as np


MEMOIZE_OPTIONS = (None, 'create', 'use')


class FeatureMapBase(object):
    """
    The is an abstract base class for creating new feature maps.

    The `__call__` member function implements the feature map computation.

    Documentation for `__call__`:

    """

    # When deriving a new feature map, these should be manually fixed
    n_local_img_features = 0
    n_local_seg_features = 0

    n_global_img_features = 0
    n_global_seg_features = 0

    def __init__(self, sigmas=[0, 3]):
        self.sigmas = sigmas
        self.nfeatures = ((self.nlocalimg + self.nglobalimg)*len(sigmas) +
                          self.nlocalseg + self.nglobalseg)

    def __call__(self, u, img, dist, mask, dx=None, memoize=None):
        if memoize not in MEMOIZE_OPTIONS:
            msg = "Memoization option ({}) was not one of {}"
            raise ValueError(msg.format(memoize, MEMOIZE_OPTIONS))

        dx = np.ones(img.ndim) if dx is None else dx

        return self.compute_features(u=u, img=img, dist=dist, mask=mask,
                                     dx=dx, memoize=memoize)

    @abc.abstractmethod
    def compute_features(self, u, img, dist, mask, dx=None, memoize=None):
        """ Compute the features

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

        dx: numpy.array, shape=img.ndim
            The "delta" spacing terms for each image axis. If None, then
            1.0 is used for each axis.

        memoize: flag, default=None
            Should be one of [None, 'create', 'use']. This variable will be False
            always during training, but during run-time (where the feature
            map is called for many iterations on the same image), it is
            'create' at the first iteration and 'use' for all iterations
            thereafter. Use this to create more efficient run-time
            performance by storing features that can be re-used in the
            first iteration which are then used in subsequent iterations.

        """
        features = np.zeros(img.shape + (self.nfeatures,))

        # Compute features ...
        # return features ...

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_features(self):
        """ The total number of features computed """
        raise NotImplementedError

    @property
    def n_local_img_features(self):
        pass

    @property
    @abc.abstractmethod
    def names(self):
        """
        Return a list of the feature names as strings. While not strictly
        necessary, you'll probably thank yourself for creating this.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def types(self):
        """
        Return a list of feature types (local or global, shape or image).
        """
        raise NotImplementedError