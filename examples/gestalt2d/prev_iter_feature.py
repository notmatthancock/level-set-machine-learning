from scipy.ndimage import gaussian_filter

from lsml.feature.base_feature import BaseShapeFeature


class PrevIterFeature(BaseShapeFeature):
    """ A feature that produces the previous level set iteration
    smoothed by a gaussian filter with parameter sigma
    """
    
    locality = 'local'

    @property
    def name(self):
        return "smooth-ls-sigma={:.2f}".format(self.sigma)

    def __init__(self, ndim=2, sigma=2):
        super().__init__(ndim)
        self.sigma = sigma

    def compute_feature(self, u, dist, mask, dx):
        return gaussian_filter(u, sigma=self.sigma)

