import unittest

from level_set_machine_learning.initialize.provided import ThresholdInitialize


class TestThresholdInitialize(unittest.TestCase):

    def test_square_image(self):

        import numpy as np

        random_state = np.random.RandomState(1234)

        mask = np.pad(np.ones((5, 5), dtype=bool), 5, 'constant')
        image = np.zeros(mask.shape)

        image[mask] = 1 + 0.5*random_state.randn(mask.sum())
        image[~mask] = -1 - 0.5*random_state.rand((~mask).sum())

        initializer = ThresholdInitialize(sigma=0)

        _, _, init_mask = initializer(image)

        import matplotlib.pyplot as pl
        pl.imshow(init_mask)
        pl.show()

        self.assertTrue((mask == init_mask).all())
