import unittest

from lsml.initializer.provided import (
    ThresholdInitializer)


class TestThresholdInitialize(unittest.TestCase):

    def test_square_image(self):

        import numpy as np

        random_state = np.random.RandomState(1234)

        mask = np.pad(np.ones((5, 5), dtype=bool), 5, 'constant')
        image = np.zeros(mask.shape)

        image[mask] = 1 + 0.5*random_state.randn(mask.sum())
        image[~mask] = -1 - 0.5*random_state.rand((~mask).sum())

        initializer = ThresholdInitializer(sigma=0)

        u0, _, _ = initializer(image)

        self.assertTrue((mask == (u0 > 0)).all())
