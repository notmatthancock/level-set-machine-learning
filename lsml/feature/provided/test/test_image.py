import unittest

import numpy as np

from lsml.feature.provided import image


class TestImageFeatures(unittest.TestCase):

    def test_image_sample2d_sigma0(self):

        random_state = np.random.RandomState(123)
        image_sample = image.ImageSample(ndim=2, sigma=0)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, mask=mask, dx=None)

        self.assertLessEqual(np.abs(feature[mask] - img[mask]).mean(), 1e-8)

    def test_image_sample2d_sigma1(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 1.0

        random_state = np.random.RandomState(123)
        image_sample = image.ImageSample(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_sample(u=u, img=img, mask=mask, dx=[1, 2])

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

        feature = image_edge_sample(u=u, img=img, mask=mask, dx=None)

        self.assertLessEqual(np.abs(feature[mask] - gmag[mask]).mean(), 1e-8)

    def test_image_edge_sample2d_sigma1(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 1

        random_state = np.random.RandomState(123)
        image_edge_sample = image.ImageEdgeSample(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        feature = image_edge_sample(u=u, img=img, mask=mask, dx=[1, 2])

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

        feature = interior_average(u=u, img=img, mask=mask)

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

        feature = interior_average(u=u, img=img, mask=mask, dx=[1, 2])

        self.assertAlmostEqual(smoothed_image[mask].mean(), feature[mask][0])

    def test_interior_image_variation_sigma0_2d(self):

        sigma = 0

        random_state = np.random.RandomState(123)
        interior_average = image.InteriorImageVariation(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        img[mask] = 2.0

        feature = interior_average(u=u, img=img, mask=mask)

        self.assertAlmostEqual(0, feature[mask][0])

    def test_interior_image_variation_sigma1_2d(self):

        from scipy.ndimage import gaussian_filter1d

        sigma = 10

        random_state = np.random.RandomState(123)
        interior_average = image.InteriorImageVariation(ndim=2, sigma=sigma)

        img = random_state.randn(100, 200)
        u = random_state.randn(*img.shape)
        mask = u > 0

        smoothed_image = gaussian_filter1d(
            gaussian_filter1d(img, sigma=sigma, axis=0),
            sigma=0.5*sigma, axis=1)

        feature = interior_average(u=u, img=img, mask=mask, dx=[1, 2])

        self.assertAlmostEqual(smoothed_image[mask].std(), feature[mask][0])

    def test_com_ray_samples2d(self):

        x, dx = np.linspace(-2, 2, 301, retstep=True)
        y, dy = np.linspace(-2, 2, 201, retstep=True)

        xx, yy = np.meshgrid(x, y)

        # Make the image a signed distance field
        img = 1 - np.sqrt(xx**2 + yy**2)

        # Set up the mask to be just a single sample
        mask = abs(img) < 0.01
        ii, jj = np.where(mask)
        mask[ii[1:], jj[1:]] = False

        # Take COM ray samples over the signed distance field.
        # This should yield samples that yield a line with slope=1
        # due to the signed distance properties (and that the center
        # of mass is at the origin).
        com_ray = image.COMRaySample(sigma=0, n_samples=10)
        features = com_ray(u=img, img=img, mask=mask, dx=[dy, dx])
        samples = features[mask][0]

        _, dt = np.linspace(-1, 1, 2*com_ray.n_samples+2, retstep=True)
        ds = np.diff(samples)

        # Difference between consecutive samples should be nearly identical
        self.assertAlmostEqual(ds.var(), 0, places=10)

        # The slope should be 1. Note that we can use the mean WLOG here
        # since the previous test asserts the differences are nearly the same
        self.assertAlmostEqual((ds/dt).mean(), 1, places=1)
