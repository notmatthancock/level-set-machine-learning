import unittest

import numpy as np

from level_set_machine_learning.feature import shape


class TestShapeFeatures(unittest.TestCase):

    def test_size_1d(self):
        """ Check interval length in 1d
        """

        x, dx = np.linspace(-2, 2, 401, retstep=True)
        y = 1 - np.abs(x)
        mask = np.zeros(x.shape, dtype=np.bool)
        mask[0] = True

        size = shape.Size(ndim=1)
        length = size(u=y, dist=y, mask=mask, dx=[dx])

        self.assertAlmostEqual(2, length[0], places=1)

    def test_size_2d(self):
        """ Check area feature with anisotropic mesh
        """
        x, dx = np.linspace(-2, 2, 701, retstep=True)
        y, dy = np.linspace(-2, 2, 901, retstep=True)

        xx, yy = np.meshgrid(x, y)

        z = 1 - np.sqrt(xx**2 + yy**2)
        mask = np.zeros(xx.shape, dtype=np.bool)
        mask[0, 0] = True

        size = shape.Size(ndim=2)
        area = size(u=z, dist=z, mask=mask, dx=[dx, dy])

        self.assertAlmostEqual(np.pi, area[0, 0], places=2)

    def test_size_3d(self):
        """ Check volume feature with anisotropic mesh
        """

        x, dx = np.linspace(-2, 2, 401, retstep=True)
        y, dy = np.linspace(-2, 2, 301, retstep=True)
        z, dz = np.linspace(-2, 2, 501, retstep=True)

        xx, yy, zz = np.meshgrid(x, y, z)

        w = 1 - np.sqrt(xx**2 + yy**2 + zz**2)
        mask = np.zeros(xx.shape, dtype=np.bool)
        mask[0, 0, 0] = True

        size = shape.Size(ndim=3)
        volume = size(u=w, dist=w, mask=mask, dx=[dx, dy, dz])

        print(volume)
        self.assertAlmostEqual(4 * np.pi / 3, volume[0, 0, 0], places=2)

    def test_boundary_1d(self):
        """ Check boundary size in 1d (# zeros)
        """
        x, dx = np.linspace(-2, 2, retstep=True)
        u = 1 - np.abs(x)
        mask = np.zeros(x.shape, dtype=np.bool)
        mask[0] = True

        boundary_size = shape.BoundarySize(ndim=1)
        n_zeros = boundary_size(u=u, dist=u, mask=mask, dx=[dx])

        self.assertAlmostEqual(2, n_zeros)

    def test_boundary_size_2d(self):
        """ Check boundary curve length with anisotropic mesh
        """
        x, dx = np.linspace(-2, 2, 501, retstep=True)
        y, dy = np.linspace(-2, 2, 701, retstep=True)

        xx, yy = np.meshgrid(x, y)

        z = 1 - np.sqrt(xx**2 + yy**2)
        mask = np.zeros(z.shape, dtype=np.bool)
        mask[0, 0] = True

        boundary_size = shape.BoundarySize(ndim=2)
        curve_length = boundary_size(u=z, dist=z, mask=mask, dx=[dx, dy])

        self.assertAlmostEqual(2*np.pi, curve_length[0, 0], places=4)

    def test_boundary_size_3d(self):
        """ Check boundary surface area with anisotropic mesh
        """
        x, dx = np.linspace(-2, 2, 501, retstep=True)
        y, dy = np.linspace(-2, 2, 401, retstep=True)
        z, dz = np.linspace(-2, 2, 601, retstep=True)

        xx, yy, zz = np.meshgrid(x, y, z)

        w = 1 - np.sqrt(xx**2 + yy**2 + zz**2)
        mask = np.zeros(w.shape, dtype=np.bool)
        mask[0, 0, 0] = True

        boundary_size = shape.BoundarySize(ndim=3)
        surface_area = boundary_size(u=w, dist=w, mask=mask, dx=[dx, dy, dz])

        self.assertAlmostEqual(4*np.pi, surface_area[0, 0, 0], places=0)

    def test_isoperimetric_2d(self):
        """ Check isoperm ratio is approx 1 for circle
        """
        x, dx = np.linspace(-2, 2, 701, retstep=True)
        y, dy = np.linspace(-2, 2, 901, retstep=True)

        xx, yy = np.meshgrid(x, y)

        z = 1 - np.sqrt(xx**2 + yy**2)
        mask = np.zeros(xx.shape, dtype=np.bool)
        mask[0, 0] = True

        isoperm = shape.IsoperimetricRatio(ndim=2)
        ratio = isoperm(u=z, dist=z, mask=mask, dx=[dx, dy])

        self.assertAlmostEqual(1, ratio[0, 0], places=3)

    def test_isoperimetric_3d(self):
        """ Check isoperm ratio is approx 1 for sphere
        """
        x, dx = np.linspace(-2, 2, 201, retstep=True)
        y, dy = np.linspace(-2, 2, 201, retstep=True)
        z, dz = np.linspace(-2, 2, 901, retstep=True)

        xx, yy, zz = np.meshgrid(x, y, z)

        w = 1 - np.sqrt(xx**2 + yy**2 + zz**2)
        mask = np.zeros(w.shape, dtype=np.bool)
        mask[0, 0, 0] = True

        isoperm = shape.IsoperimetricRatio(ndim=3)
        ratio = isoperm(u=w, dist=w, mask=mask, dx=[dx, dy, dz])

        self.assertAlmostEqual(1, ratio[0, 0, 0], places=2)

    def test_moment2d_order1(self):
        """ Check that center of mass is in the correct location
        """
        x, dx = np.linspace(-2, 2, 301, retstep=True)
        y, dy = np.linspace(-2, 2, 501, retstep=True)

        xx, yy = np.meshgrid(x, y)

        z = 1 - np.sqrt(xx**2 + yy**2)
        mask = np.zeros(xx.shape, dtype=np.bool)
        mask[0, 0] = True

        moments = [shape.Moment(ndim=2, axis=0, order=1),
                   shape.Moment(ndim=2, axis=1, order=1)]

        center_of_mass = [
            moment(u=z, dist=z, mask=mask, dx=[dy, dx])
            for moment in moments
        ]

        self.assertAlmostEqual(2.0, center_of_mass[0][mask][0], places=3)
        self.assertAlmostEqual(2.0, center_of_mass[1][mask][0], places=3)

    def test_moment2d_order1_off_center(self):
        """ Check that center of mass is in the correct location
        """
        x, dx = np.linspace(-2, 2, 301, retstep=True)
        y, dy = np.linspace(-2, 2, 501, retstep=True)

        xx, yy = np.meshgrid(x, y)

        z = 1 - np.sqrt((xx-0.25)**2 + (yy+0.25)**2)
        mask = np.zeros(xx.shape, dtype=np.bool)
        mask[0, 0] = True

        moments = [shape.Moment(ndim=2, axis=0, order=1),
                   shape.Moment(ndim=2, axis=1, order=1)]

        center_of_mass = [
            moment(u=z, dist=z, mask=mask, dx=[dy, dx])
            for moment in moments
        ]

        self.assertAlmostEqual(1.75, center_of_mass[0][mask][0], places=3)
        self.assertAlmostEqual(2.25, center_of_mass[1][mask][0], places=3)

    def test_moment2d_order2(self):
        """ Check that center of mass is in the correct location
        """
        x, dx = np.linspace(-2, 2, 301, retstep=True)
        y, dy = np.linspace(-2, 2, 201, retstep=True)

        xx, yy = np.meshgrid(x, y)

        z = 1 - np.sqrt(xx**2 + yy**2)
        mask = np.zeros(xx.shape, dtype=np.bool)
        mask[0, 0] = True

        moments = [shape.Moment(ndim=2, axis=0, order=2),
                   shape.Moment(ndim=2, axis=1, order=2)]

        spread = [
            moment(u=z, dist=z, mask=mask, dx=[dy, dx])
            for moment in moments
        ]

        self.assertAlmostEqual(0.25, spread[0][mask][0], places=2)
        self.assertAlmostEqual(0.25, spread[1][mask][0], places=2)

