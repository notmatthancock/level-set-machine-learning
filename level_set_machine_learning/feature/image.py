import numpy
from scipy.ndimage.filters import gaussian_filter1d

from .base_feature import (
    BaseImageFeature, LOCAL_FEATURE_TYPE, GLOBAL_FEATURE_TYPE)


class SmoothedImageSample(BaseImageFeature):
    """ The gaussian-smoothed image sampled locally
    """
    locality = LOCAL_FEATURE_TYPE

    def name(self):
        return "Smoothed image sample (sigma = {.3f})".format(self.sigma)

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
        super(SmoothedImageSample, self).__init__(ndim)
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

        return feature

