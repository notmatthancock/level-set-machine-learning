import unittest

from level_set_machine_learning.initialize.initialize_base import (
    InitializeBase)


class TestInitializeBase(unittest.TestCase):

    def test_call(self):

        import numpy as np

        random_state = np.random.RandomState(123)

        # Mock an initializer class
        class MyInitialize(InitializeBase):

            def initialize(self, img, dx, seed):
                return img > 0

        image = random_state.randn(41, 50)

        initializer = MyInitialize()
        u0, _, _ = initializer(image)

        self.assertTrue(((u0 > 0) == (image > 0)).all())
