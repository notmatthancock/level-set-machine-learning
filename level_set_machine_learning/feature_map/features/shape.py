import numpy

from .base_feature import (
    BaseShapeFeature, LOCAL_FEATURE_TYPE, GLOBAL_FEATURE_TYPE)


class Size(BaseShapeFeature):
    """ Computes the size of the region enclosed by the zero level set of u.
    In 1D, this is length. In 2D, it is area, and in 3D, it is volume.
    """

    locality = GLOBAL_FEATURE_TYPE

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

    def compute_feature(self, u, dist, mask, dx):

        size = (u > 0).sum() * numpy.prod(dx)
        feature = numpy.empty(u.shape)
        feature[mask] = size

        return feature


class BoundarySize(BaseShapeFeature):
    """ Computes the size of the zero-level set of u. In 2D, this is
    the arc-length of the implicit curve. In 3D, it is surface area.

    This uses a volume integral:

        :math:`\\int\\int \\delta(u) \\| Du \\| dx dy`

    to compute the length of the boundary contour via Co-Area formula.
    """

    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        if self.ndim == 1:
            return 'zeros'
        elif self.ndim == 2:
            return 'arc-length'
        elif self.ndim == 3:
            return 'surface-area'
        else:
            return 'hyper-surface-area'

    def compute_feature(self, u, dist, mask, dx):

        pos_part = (u > 0).astype(numpy.float)

        gradient = numpy.gradient(pos_part.astype(numpy.float), *dx)
        gradient_magnitude = numpy.zeros_like(gradient[0])

        for grad in gradient:
            gradient_magnitude += grad**2
        gradient_magnitude **= 0.5

        return gradient_magnitude.sum() * numpy.prod(dx)


class IsoperimetricRatio(BaseShapeFeature):
    """ Computes the isoperimetric ratio, which is a measure of
    circularity in two dimensions and a measure of sphericity in three.
    """

    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        if self.ndim == 2:
            return 'circularity'
        else:
            return 'sphericity'

    def __init__(self, ndim):
        if ndim < 2 or ndim > 3:
            msg = ("Isoperimetric ratio defined for dimensions 2 and 3; "
                   "ndim supplied = {}")
            raise ValueError(msg.format(ndim))

        super(IsoperimetricRatio, self).__init__(ndim)

    def compute_feature(self, u, dist, mask, dx):

        return gradient_magnitude.sum() * numpy.prod(dx)
