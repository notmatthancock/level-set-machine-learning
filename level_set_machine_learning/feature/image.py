from functools import reduce

import numpy
from scipy.ndimage.filters import gaussian_filter1d

from .base_feature import (
    BaseImageFeature, LOCAL_FEATURE_TYPE, GLOBAL_FEATURE_TYPE)


class ImageSample(BaseImageFeature):
    """ The gaussian-smoothed image sampled locally
    """
    locality = LOCAL_FEATURE_TYPE

    def name(self):
        return "Image sample (sigma = {.3f})".format(self.sigma)

    def __init__(self, ndim, sigma):
        """ Initialize a smoothed image sample feature

        ndim: int
            The number of dimensions in which the feature will be computed

        sigma: float
            The smooth parameter for Gaussian smoothing (note that
            sigma = 0 yields no smoothing; also note that anisotropic
            volumes will alter sigma along each axis according to the
            provided dx terms)

        """
        super(ImageSample, self).__init__(ndim)
        self.sigma = sigma

    def compute_feature(self, u, img, dist, mask, dx):
        feature = numpy.empty_like(u)

        if self.sigma == 0:
            feature[mask] = img[mask]
        else:
            smoothed = img.copy()

            for i in range(self.ndim):
                smoothed = gaussian_filter1d(
                    smoothed, sigma=self.sigma / dx[i], axis=i)

            feature[mask] = smoothed[mask]

        return feature


class ImageEdgeSample(BaseImageFeature):
    """ The gaussian-smoothed image edge (gradient magnitude) sampled locally
    """
    locality = LOCAL_FEATURE_TYPE

    def name(self):
        return "Image edge sample (sigma = {.3f})".format(self.sigma)

    def __init__(self, ndim, sigma):
        """ Initialize a smoothed image sample feature

        ndim: int
            The number of dimensions in which the feature will be computed

        sigma: float
            The smooth parameter for Gaussian smoothing (note that
            sigma = 0 yields no smoothing; also note that anisotropic
            volumes will alter sigma along each axis according to the
            provided dx terms)

        """
        super(ImageEdgeSample, self).__init__(ndim)
        self.sigma = sigma

    def compute_feature(self, u, img, dist, mask, dx):
        feature = numpy.empty_like(u)

        if self.sigma == 0:
            gradients = numpy.gradient(img, *dx)

        else:
            gradients = [
                gaussian_filter1d(
                    img, sigma=self.sigma/dx[axis], order=1, axis=axis)
                for axis in range(self.ndim)
            ]

        # Square, sum, and square-root the gradient terms to form the magnitude
        gradient_magnitude = reduce(lambda a, b: a+b**2,
                                    gradients,
                                    numpy.zeros_like(img))**0.5

        # Place the gradient magnitude into the feature array
        feature[mask] = gradient_magnitude[mask]

        return feature

