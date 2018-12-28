import unittest

import numpy as np

from level_set_machine_learning.feature import image


class TestImageFeatures(unittest.TestCase):

    def test_image_sample2d_sigma0(self):

        random_state = np.random.RandomState(123)
        image_sample = image.ImageSample(ndim=2, sigma=0)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, dist=u, mask=mask, dx=None)

        self.assertLessEqual(np.abs(feature[mask] - img[mask]).mean(), 1e-8)

    def test_image_sample2d_sigma1(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 1.0

        random_state = np.random.RandomState(123)
        image_sample = image.ImageSample(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, dist=u, mask=mask, dx=[1, 2])

        smoothed_img = gaussian_filter1d(
            gaussian_filter1d(img, sigma, axis=0), 0.5*sigma, axis=1)

        diff = np.abs(feature[mask] - smoothed_img[mask]).mean()

        self.assertLessEqual(diff, 1e-8)

    def test_image_edge_sample2d_sigma0(self):

        random_state = np.random.RandomState(123)
        image_edge_sample = image.ImageEdgeSample(ndim=2, sigma=0)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        di, dj = np.gradient(img)
        gmag = np.sqrt(di**2 + dj**2)

        feature = image_edge_sample(u=u, img=img, dist=u, mask=mask, dx=None)

        self.assertLessEqual(np.abs(feature[mask] - gmag[mask]).mean(), 1e-8)

    def test_image_edge_sample2d_sigma1(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 1

        random_state = np.random.RandomState(123)
        image_edge_sample = image.ImageEdgeSample(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_edge_sample(u=u, img=img, dist=u, mask=mask, dx=[1, 2])

        di = gaussian_filter1d(img, sigma=sigma, axis=0, order=1)
        dj = gaussian_filter1d(img, sigma=0.5*sigma, axis=1, order=1)

        gmag = np.sqrt(di**2 + dj**2)

        self.assertLessEqual(np.abs(feature[mask] - gmag[mask]).mean(), 1e-8)

    def test_interior_image_average_sigma0_2d(self):

        sigma = 0

        random_state = np.random.RandomState(123)
        interior_average = image.InteriorImageAverage(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        img[mask] = 2.0

        feature = interior_average(u=u, img=img, dist=u, mask=mask)

        self.assertAlmostEqual(2.0, feature[mask][0])

    def test_interior_image_average_sigma1_2d(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 10

        random_state = np.random.RandomState(123)
        interior_average = image.InteriorImageAverage(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        smoothed_image = gaussian_filter1d(
            gaussian_filter1d(img, sigma=sigma, axis=0),
            sigma=0.5*sigma, axis=1)

        feature = interior_average(u=u, img=img, dist=u, mask=mask, dx=[1, 2])

        self.assertAlmostEqual(smoothed_image[mask].mean(), feature[mask][0])
