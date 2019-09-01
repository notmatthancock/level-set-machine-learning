import unittest

from lsml.initializer.provided.ball import RandomBallInitializer


class TestBallInitialize(unittest.TestCase):

    def test_random_ball_smoke_test(self):

        import numpy as np

        random_state = np.random.RandomState(1234)

        mask = np.pad(np.ones((5, 5), dtype=bool), 5, 'constant')
        image = np.zeros(mask.shape)

        image[mask] = 1 + 0.5*random_state.randn(mask.sum())
        image[~mask] = -1 - 0.5*random_state.rand((~mask).sum())

        initializer = RandomBallInitializer(random_state=random_state)

        # Smoke test ...
        u0, _, _ = initializer(image)
        self.assertTrue((u0 > 0).any())
