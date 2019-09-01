import unittest

from lsml.initializer.initializer_base import InitializerBase


class TestInitializeBase(unittest.TestCase):

    def test_call(self):

        import numpy as np

        random_state = np.random.RandomState(123)

        # Mock an initializer class
        class MyInitializer(InitializerBase):

            def initialize(self, img, dx, seed):
                return img > 0

        image = random_state.randn(41, 50)

        initializer = MyInitializer()
        u0, _, _ = initializer(image)

        self.assertTrue(((u0 > 0) == (image > 0)).all())
