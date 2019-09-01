from functools import reduce

import numpy
from scipy.ndimage.filters import gaussian_filter1d

from lsml.feature.base_feature import (
    BaseImageFeature, LOCAL_FEATURE_TYPE, GLOBAL_FEATURE_TYPE)


class ImageSample(BaseImageFeature):
    """ The gaussian-smoothed image sampled locally
    """
    locality = LOCAL_FEATURE_TYPE

    @property
    def name(self):
        return "Image sample (\u03c3 = {:.3f})".format(self.sigma)

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

    @property
    def name(self):
        return "Image edge sample (\u03c3 = {:.3f})".format(self.sigma)

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


class InteriorImageAverage(BaseImageFeature):
    """ The gaussian-smoothed image average inside the segmentation boundaries
    """
    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        return "Interior image average (\u03c3 = {:.3f})".format(self.sigma)

    def compute_feature(self, u, img, dist, mask, dx):

        feature = numpy.empty_like(u)

        if self.sigma == 0:
            feature[mask] = img[mask].mean()
        else:
            smoothed = img.copy()

            for i in range(self.ndim):
                smoothed = gaussian_filter1d(
                    smoothed, sigma=self.sigma / dx[i], axis=i)

            feature[mask] = smoothed[mask].mean()

        return feature


class InteriorImageVariation(BaseImageFeature):
    """ The gaussian-smoothed image standard deviation inside the
    segmentation boundaries
    """
    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        return "Interior image variation (\u03c3 = {:.3f})".format(self.sigma)

    def compute_feature(self, u, img, dist, mask, dx):

        feature = numpy.empty_like(u)

        if self.sigma == 0:
            feature[mask] = img[mask].std()
        else:
            smoothed = img.copy()

            for i in range(self.ndim):
                smoothed = gaussian_filter1d(
                    smoothed, sigma=self.sigma / dx[i], axis=i)

            feature[mask] = smoothed[mask].std()

        return feature


def get_basic_image_features(ndim=2, sigmas=[0, 2]):
    """ Generate a list of basic image features at multiple sigma values

    Parameters
    ----------
    ndim : int, default=2
        The number of dimension of the image to which these features
        will be applied
    sigmas : list[float], default=[0, 2]
        A list of sigma values

    Returns
    -------
    features : list[BaseImageFeature]
        A list of image feature instances
    """
    feature_classes = [
        ImageSample,
        ImageEdgeSample,
        InteriorImageAverage,
        InteriorImageVariation,
    ]
    return [
        feature_class(ndim=ndim, sigma=sigma)
        for feature_class in feature_classes
        for sigma in sigmas
    ]
