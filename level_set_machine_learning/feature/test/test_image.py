import unittest

import numpy as np

from level_set_machine_learning.feature import image


class TestImageFeatures(unittest.TestCase):

    def test_image_sample2d_sigma0(self):

        random_state = np.random.RandomState(123)
        image_sample = image.SmoothedImageSample(ndim=2, sigma=0)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, dist=u, mask=mask, dx=None)

        self.assertLessEqual(np.abs(feature[mask] - img[mask]).mean(), 1e-8)

    def test_image_sample2d_sigma1(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 1.0

        random_state = np.random.RandomState(123)
        image_sample = image.SmoothedImageSample(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, dist=u, mask=mask, dx=[1, 2])

        smoothed_img = gaussian_filter1d(
            gaussian_filter1d(img, sigma, axis=0), 0.5*sigma, axis=1)

        diff = np.abs(feature[mask] - smoothed_img[mask]).mean()

        self.assertLessEqual(diff, 1e-8)
