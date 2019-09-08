import numpy
from skimage.measure import marching_cubes_lewiner as marching_cubes
from skimage.measure import find_contours, mesh_surface_area

from lsml.feature.base_feature import (
    BaseShapeFeature, GLOBAL_FEATURE_TYPE, LOCAL_FEATURE_TYPE)


class Size(BaseShapeFeature):
    """ Computes the size of the region enclosed by the zero level set of u.
    In 1D, this is length. In 2D, it is area, and in 3D, it is volume.
    """
    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        if self.ndim == 1:
            return 'Length'
        elif self.ndim == 2:
            return 'Area'
        elif self.ndim == 3:
            return 'Volume'
        else:
            return 'Hyper-volume'

    def compute_feature(self, u, dist, mask, dx):

        size = (u > 0).sum() * numpy.prod(dx)
        feature = numpy.empty_like(u)
        feature[mask] = size

        return feature


class BoundarySize(BaseShapeFeature):
    """ Computes the size of the zero-level set of u. In 2D, this is
    the length of the implicit curve. In 3D, it is surface area.
    """
    locality = GLOBAL_FEATURE_TYPE

    def __init__(self, ndim=2):
        if ndim < 2 or ndim > 3:
            msg = ("Boundary size is only defined for dimensions 2 and 3; "
                   "ndim provided = {}")
            raise ValueError(msg.format(ndim))

        super(BoundarySize, self).__init__(ndim)

    @property
    def name(self):
        if self.ndim == 2:
            return 'Curve length'
        elif self.ndim == 3:
            return 'Surface area'

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
            return 'Circularity'
        else:
            return 'Sphericity'

    def __init__(self, ndim=2):
        if ndim < 2 or ndim > 3:
            msg = ("Isoperimetric ratio defined for dimensions 2 and 3; "
                   "ndim provided = {}")
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

        feature = numpy.empty_like(u)
        feature[mask] = 4*numpy.pi*area[mask] / curve_length[mask]**2

        return feature

    def compute_feature3d(self, u, dist, mask, dx):

        # Compute the area
        size = Size(ndim=3)
        volume = size(u=u, dist=dist, mask=mask, dx=dx)

        # Compute the area
        boundary_size = BoundarySize(ndim=3)
        surface_area = boundary_size(u=u, dist=dist, mask=mask, dx=dx)

        feature = numpy.empty_like(u)
        feature[mask] = 36*numpy.pi*volume[mask]**2 / surface_area[mask]**3

        return feature


class Moments(BaseShapeFeature):
    """ Computes the normalized statistical moments of a given order along
    a given axis
    """
    locality = GLOBAL_FEATURE_TYPE

    @property
    def name(self):
        return "Moments (axes={}; orders={})".format(self.axes, self.orders)

    @property
    def size(self):
        return len(self.axes) * len(self.orders)

    def __init__(self, ndim=2, axes=(0, 1), orders=(1, 2)):
        """ Initialize a normalized statistical moment feature

        ndim: int
            Number of dimensions

        axes: list[int], default=[0, 1]
            The axes along which the moment should be computed

        order: list[int], default=[1, 2]
            The orders of the moments, e.g., order=1 yields the 'center of
            mass' coordinate along the given axis and order=2 yields a measure
            of spread along the given axis

        """

        super(Moments, self).__init__(ndim)

        for axis in axes:
            if axis < 0 or axis > ndim-1:
                msg = "axis provided ({}) must be one of 0 ... {}"
                raise ValueError(msg.format(axis, ndim-1))

        for order in orders:
            if order < 1:
                msg = "Moments order should be greater than or equal to 1"
                raise ValueError(msg)

        self.axes = axes
        self.orders = orders

    def _compute_center_of_mass(self, u, dx):

        # Initialize center of mass container and mask with singular entry
        center_of_mass = numpy.zeros(self.ndim)
        mask = numpy.empty(u.shape, dtype=numpy.bool)
        mask.ravel()[0] = True

        for i in range(self.ndim):
            center_of_mass[i] = self._compute_moment(
                u=u, dist=u, mask=mask, dx=dx, axis=i, order=1)

        return center_of_mass

    def _compute_moment(self, u, dist, mask, dx, axis, order):
        """ Computes the feature for just a single axis and order """

        indices = numpy.indices(u.shape, dtype=numpy.float)
        mesh = indices[axis] * dx[axis]

        size = Size(ndim=self.ndim)

        # Normalize by centering if order is greater than 1
        if order > 1:
            center_of_mass = self._compute_center_of_mass(u=u, dx=dx)
            mesh -= center_of_mass[axis]

        measure = size(u=u, dist=dist, mask=mask, dx=dx)[mask].ravel()[0]
        moment = (mesh**order)[u > 0].sum() * numpy.prod(dx) / measure

        return moment

    def compute_feature(self, u, dist, mask, dx):
        from itertools import product
        features = numpy.empty(u.shape + (self.size,))
        for i, (axis, order) in enumerate(product(self.axes, self.orders)):
            features[mask, i] = self._compute_moment(
                u, dist, mask, dx, axis, order)
        return features


class DistanceToCenterOfMass(BaseShapeFeature):
    """ Computes the distance to the computed center of mass
    """
    locality = LOCAL_FEATURE_TYPE

    @property
    def name(self):
        return "Distance to center of mass"

    def compute_feature(self, u, dist, mask, dx):

        # Sneakily use the center of mass utility buried in the
        # moment feature class
        moment_feature = Moments(ndim=self.ndim)
        center_of_mass = moment_feature._compute_center_of_mass(u, dx)

        # Add extra axes for some broadcasting below
        slicer = tuple([slice(None), ] + [None for _ in range(self.ndim)])

        indices = numpy.indices(u.shape, dtype=numpy.float)
        mesh = indices * dx[slicer]

        feature = numpy.empty_like(u)
        feature[mask] = numpy.linalg.norm(
            mesh - center_of_mass[slicer], axis=0)[mask]

        return feature


def get_basic_shape_features(ndim=2, moment_orders=[1, 2]):
    """ Generate a list of basic shape features at multiple sigma values

    Parameters
    ----------
    ndim : int, default=2
        The number of dimension of the image to which these features
        will be applied
    moment_orders : list[float], default=[1, 2]
        Orders for which we compute moments

    Returns
    -------
    features : list[BaseImageFeature]
        A list of image feature instances
    """
    feature_classes = [
        BoundarySize,
        DistanceToCenterOfMass,
        IsoperimetricRatio,
        Size,
    ]
    features = [
        feature_class(ndim=ndim)
        for feature_class in feature_classes
    ]
    for axis in range(ndim):
        for order in moment_orders:
            features.append(Moments(ndim=ndim, axis=axis, order=order))
    return features
