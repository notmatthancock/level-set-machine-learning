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
    the -length of the implicit curve. In 3D, it is surface area.

    This uses a volume integral:

        :math:`\\int \\| DH(u) \\| dx `

    to compute the length of the boundary contour via Co-Area formula.
    """

    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        if self.ndim == 1:
            return 'number-zeros'
        elif self.ndim == 2:
            return 'curve-length'
        elif self.ndim == 3:
            return 'surface-area'
        else:
            return 'hyper-surface-area'

    def compute_feature(self, u, dist, mask, dx):

        positive_part = (u > 0).astype(numpy.float)

        gradient = numpy.gradient(positive_part.astype(numpy.float), *dx)
        if self.ndim == 1:
            gradient = [gradient]
        gradient_magnitude = numpy.zeros_like(u)

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

        if self.ndim == 2:
            return self.compute_feature2d(
                u=u, dist=dist, mask=mask, dx=dx)
        else:
            return self.compute_feature3d(
                u=u, dist=dist, mask=mask, dx=dx)

    def compute_feature2d(self, u, dist, mask, dx):

        # Compute the area
        size = Size(ndim=2)
        area = size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        # Compute the area
        boundary_size = BoundarySize(ndim=2)
        curve_length = boundary_size.compute_feature(
            u=u, dist=dist, mask=mask, dx=dx)

        print(area, curve_length)

        return 4*numpy.pi*area / curve_length**2

    def compute_feature3d(self, u, dist, mask, dx):

        # Compute the area
        size = Size(ndim=3)
        volume = size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        # Compute the area
        boundary_size = BoundarySize(ndim=3)
        surface_area = boundary_size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        return 36*numpy.pi*volume**2 / surface_area**3


class Moment(BaseShapeFeature):
    """ Computes the statisical moments of a given order along a given axis
    """

    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        return "moment-axis-{}-order-{}".format(self.axis, self.order)

    def __init__(self, ndim, axis=0, order=1):

        super(Moment, self).__init__(ndim)

        if axis < 0 or axis > ndim-1:
            msg = "axis provided ({}) does not lie in required range (0-{})"
            raise ValueError(msg.format(axis, ndim-1))

        if order < 1:
            raise ValueError("Moment order should be ≥ 1")

        self.axis = axis
        self.order = order

    def compute_feature(self, u, dist, mask, dx):

        if self.ndim == 2:
            return self.compute_feature2d(
                u=u, dist=dist, mask=mask, dx=dx)
        else:
            return self.compute_feature3d(
                u=u, dist=dist, mask=mask, dx=dx)

    def compute_feature2d(self, u, dist, mask, dx):

        # Compute the area
        size = Size(ndim=2)
        area = size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        # Compute the area
        boundary_size = BoundarySize(ndim=2)
        curve_length = boundary_size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        return 4 * numpy.pi * area / curve_length**2

    def compute_feature3d(self, u, dist, mask, dx):

        # Compute the area
        size = Size(ndim=3)
        volume = size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        # Compute the area
        boundary_size = BoundarySize(ndim=3)
        surface_area = boundary_size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)


