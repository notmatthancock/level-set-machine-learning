import unittest

import numpy as np

from level_set_machine_learning.feature.base_feature import (
    BaseImageFeature, BaseShapeFeature)


class TestBaseFeature(unittest.TestCase):

    def setUp(self):

        class SomeImageFeature(BaseImageFeature):
            """ Mock an image feature class """

            name = None
            locality = None

            def compute_feature(self, u, img, dist, mask, dx):
                pass

        class SomeShapeFeature(BaseShapeFeature):
            """ Mock a shape feature class """

            name = None
            locality = None

            def compute_feature(self, u, dist, mask, dx):
                pass

        self.ImageFeature = lambda ndim: SomeImageFeature(ndim=ndim)
        self.ShapeFeature = lambda ndim: SomeShapeFeature(ndim=ndim)

    def test_correct_type(self):

        from level_set_machine_learning.feature.base_feature import (
            IMAGE_FEATURE_TYPE, SHAPE_FEATURE_TYPE)

        image_feature = self.ImageFeature(ndim=1)
        shape_feature = self.ShapeFeature(ndim=1)

        self.assertEqual(image_feature.type, IMAGE_FEATURE_TYPE)
        self.assertEqual(shape_feature.type, SHAPE_FEATURE_TYPE)

    def test_mismatch_input_shapes(self):

        image_feature = self.ImageFeature(ndim=2)
        shape_feature = self.ShapeFeature(ndim=2)

        u = np.ones((3, 2))
        img = np.ones((3, 2))
        dist = np.ones((3, 1))
        mask = np.ones((3, 2), dtype=np.bool)

        with self.assertRaises(ValueError):
            image_feature(u=u, img=img, dist=dist, mask=mask)

        with self.assertRaises(ValueError):
            shape_feature(u=u, dist=dist, mask=mask)

    def test_mismatch_input_ndims(self):

        image_feature = self.ImageFeature(ndim=2)
        shape_feature = self.ShapeFeature(ndim=2)

        u = np.ones((3, 2))
        img = np.ones((3, 2))
        dist = np.ones((3,))
        mask = np.ones((3, 2), dtype=np.bool)

        with self.assertRaises(ValueError):
            image_feature(u=u, img=img, dist=dist, mask=mask)

        with self.assertRaises(ValueError):
            shape_feature(u=u, dist=dist, mask=mask)

    def test_mismatch_delta_terms(self):

        image_feature = self.ImageFeature(ndim=2)
        shape_feature = self.ShapeFeature(ndim=2)

        u = np.ones((3, 2))
        img = np.ones((3, 2))
        dist = np.ones((3, 2))
        mask = np.ones((3, 2), dtype=np.bool)

        # ndim = 2, but provide only 1 dx term
        dx = [1]

        with self.assertRaises(ValueError):
            image_feature(u=u, img=img, dist=dist, mask=mask, dx=dx)

        with self.assertRaises(ValueError):
            shape_feature(u=u, dist=dist, mask=mask, dx=dx)

    def test_feature_eq(self):

        image_feature1 = self.ImageFeature(ndim=2)
        image_feature2 = self.ImageFeature(ndim=2)

        shape_feature1 = self.ShapeFeature(ndim=2)
        shape_feature2 = self.ShapeFeature(ndim=2)

        self.assertEqual(image_feature1, image_feature2)
        self.assertEqual(shape_feature1, shape_feature2)

    def test_feature_neq(self):

        image_feature1 = self.ImageFeature(ndim=2)
        image_feature2 = self.ImageFeature(ndim=2)
        image_feature1.name = 'other image feature'

        shape_feature1 = self.ShapeFeature(ndim=2)
        shape_feature2 = self.ShapeFeature(ndim=2)
        shape_feature1.name = 'other shape feature'

        self.assertNotEqual(image_feature1, image_feature2)
        self.assertNotEqual(shape_feature1, shape_feature2)

    def test_feature_hash(self):

        image_feature1 = self.ImageFeature(ndim=2)
        image_feature2 = self.ImageFeature(ndim=2)

        shape_feature1 = self.ShapeFeature(ndim=2)
        shape_feature2 = self.ShapeFeature(ndim=2)
        shape_feature1.name = 'other shape feature'

        image_feature_set = {image_feature1, image_feature2}
        shape_feature_set = {shape_feature1, shape_feature2}

        self.assertEqual(1, len(image_feature_set))
        self.assertEqual(2, len(shape_feature_set))

    def test_empty_mask(self):

        image_feature = self.ImageFeature(ndim=2)
        shape_feature = self.ShapeFeature(ndim=2)

        u = np.ones((3, 2))
        img = np.ones((3, 2))
        dist = np.ones((3, 2))
        mask = np.zeros((3, 2), dtype=np.bool)

        feature = image_feature(u=u, img=img, dist=dist, mask=mask)
        self.assertEqual(u.shape, feature.shape)
        self.assertEqual(u.dtype, feature.dtype)

        feature = shape_feature(u=u, dist=dist, mask=mask)
        self.assertEqual(u.shape, feature.shape)
        self.assertEqual(u.dtype, feature.dtype)
