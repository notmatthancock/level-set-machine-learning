from skimage.feature.corner import corner_harris
from scipy.ndimage import gaussian_filter

from lsml.feature.base_feature import BaseImageFeature


class CornerFeature(BaseImageFeature):

    locality = 'local'

    @property
    def name(self):
        return "corner-reponse-sigma={:.2f}".format(self.sigma)

    def __init__(self, ndim=2, sigma=2):
        super().__init__(ndim)
        self.sigma = sigma

    def compute_feature(self, u, img, dist, mask, dx):
        return corner_harris(gaussian_filter(img, sigma=self.sigma))
