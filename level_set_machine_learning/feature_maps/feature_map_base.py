"""
Abstract base class for feature map functions.

Use this class is a "template" for creating feature map classes.

For a specific example, see the `dim2/simple_feature_map.py` class.
"""
import abc

import numpy as np


MEMOIZE_OPTIONS = (None, 'create', 'use')


class FeatureMapBase(abc.ABC):
    """
    The is a template class for creating new feature maps.

    The `__call__` member function implements the feature map computation.

    Documentation for `__call__`:

    Parameters
    ----------
    u: ndarrary
        The "level set function".

    img: ndarray
        The image/image volume/image hypervolume.

    dist: ndarray
        The signed distance transform to the level set `u` (only computed
        necessarily in the narrow band region).

    mask: ndarray, dtype=bool
        The boolean mask indicating the narrow band region of `u`.

    dx: ndarray, shape=img.ndim
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
    # These should be set manual according to the specific feature map.
    nlocalimg  = 0
    nlocalseg  = 0
    nglobalimg = 0
    nglobalseg = 0

    def __init__(self, sigmas=[0, 3]):
        self.sigmas = sigmas
        self.nfeatures = ((self.nlocalimg + self.nglobalimg)*len(sigmas) +
                          self.nlocalseg + self.nglobalseg)

    @abc.abstractmethod
    def __call__(self, u, img, dist, mask, dx=None, memoize=None):
        if memoize not in MEMOIZE_OPTIONS:
            msg = "Memoization option ({}) was not one of {}"
            raise ValueError(msg.format(memoize, MEMOIZE_OPTIONS))

        dx = np.ones(img.ndim) if dx is None else dx

        features = np.zeros(img.shape + (self.nfeatures,))

        # Compute features ...

        return features

    @property
    @abc.abstractmethod
    def names(self):
        """
        Return a list of the feature names as strings. While not strictly
        necessary, you'll probably thank yourself for creating this.
        """
        raise NotImplementedError

    @property
    def types(self):
        """
        Return a list of feature types (local or global, shape or image).
        """
        raise NotImplementedError