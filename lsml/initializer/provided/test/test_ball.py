import unittest

import numpy as np

from lsml.initializer.provided.ball import (
    BallInitializer, RandomBallInitializer)


class TestBallInitialize(unittest.TestCase):

    def test_centered_ball2d(self):
        shape = (51, 51)
        img = np.zeros(shape)

        radius = 10
        initializer = BallInitializer(radius=radius)

        u, _, _ = initializer(img)

        diameter = (u[shape[0]//2] > 0).sum()

        self.assertEqual(radius, diameter // 2)

    def test_centered_ball3d(self):
        shape = (31, 31, 31)
        img = np.zeros(shape)

        radius = 10
        initializer = BallInitializer(radius=radius)

        u, _, _ = initializer(img)
        diameter = (u[shape[0]//2, shape[1]//2] > 0).sum()

        self.assertEqual(radius, diameter // 2)

    def test_centered_anisotropic_ball2d(self):
        shape = (51, 51)
        img = np.zeros(shape)

        radius = 10
        initializer = BallInitializer(radius=radius)

        dx = [2, 0.5]
        u, _, _ = initializer(img, dx=dx)

        # The dx along axis 0 is 2, so we expect the
        # diameter in index space to be squashed by a factor of 2^-1
        factor = 2**-1
        diameter_axis0 = (u[:, 25] > 0).sum()
        self.assertEqual(radius*factor, diameter_axis0 // 2)

        # The dx along axis 1 is 0.5, so we expect the
        # diameter in index space to be squashed by a factor of 0.5^-1
        factor = 0.5**-1
        diameter_axis1 = (u[25, :] > 0).sum()
        self.assertEqual(radius*factor, diameter_axis1 // 2)

    def test_random_ball_smoke_test(self):

        random_state = np.random.RandomState(1234)

        mask = np.pad(np.ones((5, 5), dtype=bool), 5, 'constant')
        image = np.zeros(mask.shape)

        image[mask] = 1 + 0.5*random_state.randn(mask.sum())
        image[~mask] = -1 - 0.5*random_state.rand((~mask).sum())

        initializer = RandomBallInitializer(random_state=random_state)

        # Smoke test ...
        u0, _, _ = initializer(image)
        self.assertTrue((u0 > 0).any())
