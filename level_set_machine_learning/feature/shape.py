import numpy
from skimage.measure import marching_cubes_lewiner as marching_cubes
from skimage.measure import find_contours, mesh_surface_area

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
        feature = numpy.empty_like(u)
        feature[mask] = size

        return feature


class BoundarySize(BaseShapeFeature):
    """ Computes the size of the zero-level set of u. In 2D, this is
    the -length of the implicit curve. In 3D, it is surface area.
    """
    locality = GLOBAL_FEATURE_TYPE

    def __init__(self, ndim):
        if ndim < 2 or ndim > 3:
            msg = ("Isoperimetric ratio defined for dimensions 2 and 3; "
                   "ndim supplied = {}")
            raise ValueError(msg.format(ndim))

        super(BoundarySize, self).__init__(ndim)

    @property
    def name(self):
        if self.ndim == 2:
            return 'curve-length'
        elif self.ndim == 3:
            return 'surface-area'

    def compute_feature(self, u, dist, mask, dx):

        feature = numpy.empty_like(u)

        if self.ndim == 2:
            boundary_size = self._compute_arc_length(u, dx)
        elif self.ndim == 3:
            boundary_size = self._compute_surface_area(u, dx)
        else:
            msg = "Cannot compute boundary size for ndim = {}"
            raise RuntimeError(msg.format(self.ndim))

        feature[mask] = boundary_size

        return feature

    def _compute_arc_length(self, u, dx):

        contours = find_contours(u, 0)

        total_arc_length = 0.

        for contour in contours:
            closed_contour = numpy.vstack((contour, contour[0]))
            closed_contour *= dx[::-1]  # find_contours points in index space
            arc_length = numpy.linalg.norm(numpy.diff(closed_contour, axis=0),
                                           axis=1).sum()
            total_arc_length += arc_length

        return total_arc_length

    def _compute_surface_area(self, u, dx):
        verts, faces, _, _ = marching_cubes(u, 0., spacing=dx)
        return mesh_surface_area(verts, faces)


class IsoperimetricRatio(BaseShapeFeature):
    """ Computes the isoperimetric ratio, which is a measure of
    circularity in two dimensions and a measure of sphericity in three.
    In both cases, the maximum ratio value of 1 is achieved only for
    a perfect circle or sphere.
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

        return 4*numpy.pi*area / curve_length**2

    def compute_feature3d(self, u, dist, mask, dx):

        # Compute the area
        size = Size(ndim=3)
        volume = size.compute_feature(u=u, dist=dist, mask=mask, dx=dx)

        # Compute the area
        boundary_size = BoundarySize(ndim=3)
        surface_area = boundary_size.compute_feature(
            u=u, dist=dist, mask=mask, dx=dx)

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
            msg = "Moment order should be greater than or equal to 1"
            raise ValueError(msg)

        self.axis = axis
        self.order = order

    def compute_feature(self, u, dist, mask, dx):
        # TODO
        pass

