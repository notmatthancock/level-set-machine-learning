import unittest

import numpy as np

from level_set_machine_learning.util.distance_transform import (
    distance_transform)


class TestDistanceTransform(unittest.TestCase):

    def test_distance_transform(self):

        arr = np.r_[-1, -1, 1, -1, -1.]
        dist, mask = distance_transform(arr, band=1, dx=[1.])

        true_dist = np.r_[0., -0.5, 0.5, -0.5, 0.]
        true_mask = np.r_[False, True, True, True, False]

        # Floating point comparison is okay here, numbers are 0, 0.5, 1, etc
        self.assertTrue((dist == true_dist).all())
        self.assertTrue((mask == true_mask).all())

    def test_all_positive(self):

        arr = np.ones((4, 5, 6))
        dist, mask = distance_transform(arr, band=0, dx=[1, 1, 1.])

        self.assertEqual(0, mask.sum())
        self.assertTrue((dist == np.inf).all())

    def test_all_negative(self):

        arr = -np.ones((4, 5, 6))
        dist, mask = distance_transform(arr, band=0, dx=[1, 1, 1.])

        self.assertEqual(0, mask.sum())
        self.assertTrue((dist == -np.inf).all())

    def test_band_zero(self):

        arr = np.ones((4, 5, 6))
        arr[2] = -1
        dist, mask = distance_transform(arr, band=0, dx=[1, 1, 1.])

        self.assertEqual(arr.size, mask.sum())

    def test_input_zero(self):

        arr = np.zeros((4, 5, 6))
        dist, mask = distance_transform(arr, band=0, dx=[1, 1, 1.])

        self.assertEqual(arr.size, mask.sum())
        self.assertTrue((dist == 0).all())
