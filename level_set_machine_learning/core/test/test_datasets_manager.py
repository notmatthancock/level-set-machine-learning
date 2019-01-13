import os
import unittest

import numpy as np

from level_set_machine_learning.core.datasets_manager import DatasetsManager


class TestDatasetsManager(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(1234)

    def test_no_data_at_init(self):

        with self.assertRaises(ValueError):
            # If the provided h5 file doesn't exist, then data must be supplied
            DatasetsManager(h5_file='this file does not exist')

    def test_data_size_mismatches(self):

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim))
            for _ in range(n_examples)
        ]

        # Create some fake segmentation data
        segs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim)) > 0
            for _ in range(n_examples)
        ]

        h5_file = 'tmp.h5'

        try:
            with self.assertRaises(ValueError):
                DatasetsManager(h5_file=h5_file, imgs=imgs, segs=segs)
        finally:
            # If the test was successful and the exception was raised,
            # then this file should not have been created; but, if the test
            # failed, it may have been created and should be deleted.
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_data_wrong_img_dtype(self):

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim)) > 0
            for _ in range(n_examples)
        ]

        # Create some fake segmentation data
        segs = [
            imgs[i] > 0
            for i in range(n_examples)
        ]

        h5_file = 'tmp.h5'

        try:
            with self.assertRaises(TypeError):
                DatasetsManager(h5_file=h5_file, imgs=imgs, segs=segs)
        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_data_wrong_seg_dtype(self):

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim))
            for _ in range(n_examples)
        ]

        # Create some fake segmentation data
        segs = [
            imgs[i]
            for i in range(n_examples)
        ]

        h5_file = 'tmp.h5'

        try:
            with self.assertRaises(TypeError):
                DatasetsManager(h5_file=h5_file, imgs=imgs, segs=segs)
        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_wrong_img_ndim(self):

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim))
            for _ in range(n_examples)
        ]

        imgs[0] = self.random_state.randn(
            *self.random_state.randint(10, 41, size=n_dim-1))

        # Create some fake segmentation data
        segs = [
            imgs[i] > 0
            for i in range(n_examples)
        ]

        h5_file = 'tmp.h5'

        try:
            with self.assertRaises(ValueError):
                DatasetsManager(h5_file=h5_file, imgs=imgs, segs=segs)
        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_wrong_seg_ndim(self):

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim))
            for _ in range(n_examples)
        ]

        # Create some fake segmentation data
        segs = [
            imgs[i] > 0
            for i in range(n_examples)
        ]

        segs[0] = np.ones((4,), dtype=np.bool)

        h5_file = 'tmp.h5'

        try:
            with self.assertRaises(ValueError):
                DatasetsManager(h5_file=h5_file, imgs=imgs, segs=segs)
        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_wrong_dx_shape(self):

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim))
            for _ in range(n_examples)
        ]

        # Create some fake segmentation data
        segs = [
            imgs[i] > 0
            for i in range(n_examples)
        ]

        h5_file = 'tmp.h5'

        dx = self.random_state.rand(n_examples, n_dim+1)

        try:
            with self.assertRaises(ValueError):
                DatasetsManager(h5_file=h5_file, imgs=imgs, segs=segs, dx=dx)
        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_convert_to_hdf5_valid(self):

        import skfmm

        n_examples = 3
        n_dim = 3

        # Create some fake image data
        imgs = [
            self.random_state.randn(
                *self.random_state.randint(10, 41, size=n_dim))
            for _ in range(n_examples)
        ]

        # Create some fake segmentation data
        segs = [
            imgs[i] > 0
            for i in range(n_examples)
        ]

        h5_file = 'tmp.h5'

        dx = self.random_state.rand(n_examples, n_dim)

        try:
            datasets_mgmt = DatasetsManager(
                h5_file=h5_file, imgs=imgs, segs=segs, dx=dx)

            for example in datasets_mgmt.iter_examples():
                index = example.index
                # Manually compute the distance transform
                dist = skfmm.distance(2*segs[index].astype(float)-1,
                                      dx=dx[index])

                # Assert image integrity
                self.assertLess(
                    np.linalg.norm(imgs[index] - example.img), 1e-8)

                # Assert segmentation integrity
                self.assertEqual(0, (segs[index] != example.seg).sum())

                # Assert distance transform integrity
                self.assertLess(
                    np.linalg.norm(dist - example.dist), 1e-8)

                # Assert delta term integrity
                self.assertLess(np.linalg.norm(dx[index] - example.dx), 1e-8)

        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)
