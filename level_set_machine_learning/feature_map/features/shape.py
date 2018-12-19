import numpy

from .base_feature import (
    BaseFeature, LOCAL_SHAPE_FEATURE_TYPE, GLOBAL_SHAPE_FEATURE_TYPE)


class Size(BaseFeature):
    """ Yields the size of the region enclosed by the zero level set of u.
    In 1D, this is length. In 2D, it is area, and in 3D, it is volume.
    """

    type = GLOBAL_SHAPE_FEATURE_TYPE

    @property
    def name(self):
        if self.ndim == 1:
            return 'length'
        elif self.ndim == 2:
            return 'area'
        elif self.ndim == 3:
            return 'volume'
        else:
            return 'hyper-volume'

    def compute_feature(self, u, img, dist, mask, dx):

        size = (u > 0).sum() * numpy.prod(dx)
        feature = numpy.empty(u.shape)
        feature[mask] = size

        return feature


class Boundary(BaseFeature):
    """ Computes the size of the zero-level set of u. In 2D, this is
    the arc-length of the implicit curve. In 3D, it is surface area.
    """
    type = GLOBAL_SHAPE_FEATURE_TYPE

    @property
    def name(self):
        if self.ndim == 1:
            return 'length'
        elif self.ndim == 2:
            return 'area'
        elif self.ndim == 3:
            return 'volume'
        else:
            return 'hyper-volume'

    def compute_feature(self, u, img, dist, mask, dx):

        H = (u > 0).astype(numpy.float)

        gradient = numpy.gradient(H.astype(numpy.float), *dx)
        gradient_magnitude = numpy.zeros_like(gradient[0])
        gmagH = [grad**2]

        # Boundary length
        # This uses a volume integral:
        # :math:`\\int\\int \\delta(u) \\| Du \\| dx dy`
        # to compute the length of the boundary contour via Co-Area formula.
        L = gmagH.sum() * pdx
        F[mask,1] = L
