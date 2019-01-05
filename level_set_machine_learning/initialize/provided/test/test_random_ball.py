import unittest

from level_set_machine_learning.initialize.provided import RandomBallInitialize


class TestRandomBallInitialize(unittest.TestCase):

    def test_random_ball_smoke_test(self):

        import numpy as np

        random_state = np.random.RandomState(1234)

        mask = np.pad(np.ones((5, 5), dtype=bool), 5, 'constant')
        image = np.zeros(mask.shape)

        image[mask] = 1 + 0.5*random_state.randn(mask.sum())
        image[~mask] = -1 - 0.5*random_state.rand((~mask).sum())

        initializer = RandomBallInitialize(random_state=random_state)

        init_mask = initializer.initialize(image)

        import matplotlib.pyplot as plt
        plt.imshow(init_mask)
        plt.show()
