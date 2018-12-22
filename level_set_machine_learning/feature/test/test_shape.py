import unittest

import numpy as np

from level_set_machine_learning.feature import shape


class TestShapeFeatures(unittest.TestCase):

    def test_size_1d(self):
        """ Check interval length in 1d """

        x, dx = np.linspace(-2, 2, 401, retstep=True)
        y = 1 - np.abs(x)
        mask = np.zeros(x.shape, dtype=np.bool)
        mask[0] = True

        size = shape.Size(ndim=1)
        length = size(u=y, dist=y, mask=mask, dx=[dx])

        self.assertAlmostEqual(2, length[0], places=1)

    def test_size_2d(self):
        """ Check area feature with anisotropic mesh """

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
        """ Check area feature with anisotropic mesh """

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
