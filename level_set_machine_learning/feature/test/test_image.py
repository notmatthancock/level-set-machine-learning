import unittest

import numpy as np

from level_set_machine_learning.feature import image


class TestShapeFeatures(unittest.TestCase):

    def test_image_sample2d(self):

        random_state = np.random.RandomState(123)
        image_sample = image.SmoothedImageSample(ndim=2, sigma=0)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, dist=u, mask=mask, dx=None)

        self.assertLessEqual(np.abs(feature[mask] - img[mask]).mean(), 1e-8)
