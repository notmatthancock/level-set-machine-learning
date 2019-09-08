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


class COMRaySample(BaseImageFeature):
    """ The image is sampled along a line segment connecting a pixel
    to the center of mass (COM) of the current segmentation iterate. Samples
    are also taken symmetrically in the outward direction.
    """
    locality = LOCAL_FEATURE_TYPE

    @property
    def name(self):
        return f"COM Ray (N={self.n_samples}, \u03c3 = {self.sigma:.3f})"

    @property
    def size(self):
        return 2*self.n_samples

    def __init__(self, ndim=2, sigma=3, n_samples=10):
        if ndim not in (2, 3):
            raise ValueError("`ndim` must be either 2 or 3")
        super().__init__(ndim, sigma)
        self.n_samples = n_samples

    def compute_feature(self, u, img, dist, mask, dx):
        from scipy.interpolate import RegularGridInterpolator
        from lsml.feature.provided.shape import Moments

        moments = Moments(ndim=self.ndim,
                          axes=list(range(self.ndim)),
                          orders=[1])
        com = moments._compute_center_of_mass(u, dx)
        t = numpy.linspace(-1, 1, 2*self.n_samples+2)[1:-1, None]

        features = numpy.empty(u.shape + (2*self.n_samples,))

        interpolator = RegularGridInterpolator(
            points=[numpy.arange(s, dtype=numpy.float)*delta
                    for s, delta in zip(u.shape, dx)],
            values=img,
            bounds_error=False,
            fill_value=0.0
        )

        coords = numpy.array(numpy.where(mask)).T

        for coord in coords:
            points = t * com[None] + (1-t) * (coord*dx)[None]
            features[tuple(coord)] = interpolator(points)

        return features


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
