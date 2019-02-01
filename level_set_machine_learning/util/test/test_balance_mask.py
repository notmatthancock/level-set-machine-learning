import unittest

import numpy

from level_set_machine_learning.util.balance_mask import balance_mask


class TestBalanceMask(unittest.TestCase):

    def test_random(self):

        random_state = numpy.random.RandomState(1234)
        arr = random_state.randn(100)
        mask = balance_mask(arr, random_state)

        self.assertEqual(len(arr[mask] > 0), len(arr[mask] < 0))

    def test_random_with_zeros(self):

        random_state = numpy.random.RandomState(1234)
        arr = random_state.randn(100)
        arr[random_state.rand(arr.shape[0]) < 0.25] = 0.

        mask = balance_mask(arr, random_state)

        self.assertEqual(len(arr[mask] > 0), len(arr[mask] < 0))

    def test_all_pos(self):

        arr = numpy.ones(100)

        random_state = numpy.random.RandomState(1234)
        mask = balance_mask(arr, random_state)

        self.assertEqual(arr.shape[0], mask.sum())

    def test_all_neg(self):

        arr = -numpy.ones(100)

        random_state = numpy.random.RandomState(1234)
        mask = balance_mask(arr, random_state)

        self.assertEqual(arr.shape[0], mask.sum())
