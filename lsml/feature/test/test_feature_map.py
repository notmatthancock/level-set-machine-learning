import unittest

import numpy as np

from lsml.feature.feature_map import FeatureMap


class TestFeatureMap(unittest.TestCase):

    def test_simple_feature_map(self):

        from lsml.feature.provided import image
        from lsml.feature.provided import shape

        features = [
            image.ImageSample(ndim=2, sigma=0),
            image.ImageEdgeSample(ndim=2, sigma=3),
            shape.Size(ndim=2),
            shape.IsoperimetricRatio(ndim=2),
            shape.Moments(ndim=2, axes=[0], orders=[1]),
        ]

        feature_map = FeatureMap(features=features)

        random_state = np.random.RandomState(1234)
        img = random_state.randn(34, 67)
        u = random_state.randn(34, 67)
        mask = random_state.randn(34, 67) > 0

        # Smoke test
        feature_map(u=u, img=img, dist=u, mask=mask)
