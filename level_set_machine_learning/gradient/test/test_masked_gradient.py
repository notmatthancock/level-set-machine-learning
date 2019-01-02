import unittest

import numpy as np

from level_set_machine_learning.gradient import masked_gradient as mg


class TestMaskedGradient(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(1234)

    def test_gradient_centered(self):

        for ndim in [1, 2, 3]:
            dims = self.random_state.randint(50, 101, size=ndim)
            arr = self.random_state.randn(*dims)

            grads, gmag = mg.gradient_centered(arr)
            numpy_grads = np.gradient(arr)
            numpy_grads = [numpy_grads] if ndim == 1 else numpy_grads

            for i in range(ndim):
                grad_error = np.abs(grads[i] - numpy_grads[i]).mean()
                self.assertLessEqual(grad_error, 1e-8)

            numpy_gmag = np.linalg.norm(numpy_grads, axis=0)
            gmag_error = np.abs(gmag - numpy_gmag).mean()
            self.assertLessEqual(gmag_error, 1e-8)

    def test_gradient_centered_aniso(self):

        for ndim in [1, 2, 3]:
            dims = self.random_state.randint(50, 101, size=ndim)
            arr = self.random_state.randn(*dims)
            dx = self.random_state.rand(ndim)

            grads, gmag = mg.gradient_centered(arr, dx=dx)
            numpy_grads = np.gradient(arr, *dx)
            numpy_grads = [numpy_grads] if ndim == 1 else numpy_grads

            for i in range(ndim):
                grad_error = np.abs(grads[i] - numpy_grads[i]).mean()
                self.assertLessEqual(grad_error, 1e-8)

            numpy_gmag = np.linalg.norm(numpy_grads, axis=0)
            gmag_error = np.abs(gmag - numpy_gmag).mean()
            self.assertLessEqual(gmag_error, 1e-8)

    def test_gradient_centered_with_mask(self):

        for ndim in [1, 2, 3]:
            dims = self.random_state.randint(50, 101, size=ndim)
            arr = self.random_state.randn(*dims)
            mask = arr > 0

            grads, gmag = mg.gradient_centered(arr, mask=mask)
            numpy_grads = np.gradient(arr)
            numpy_grads = [numpy_grads] if ndim == 1 else numpy_grads

            for i in range(ndim):
                grad_error = np.abs(grads[i] - numpy_grads[i])[mask].mean()
                self.assertLessEqual(grad_error, 1e-8)

            numpy_gmag = np.linalg.norm(numpy_grads, axis=0)
            gmag_error = np.abs(gmag - numpy_gmag)[mask].mean()
            self.assertLessEqual(gmag_error, 1e-8)

    def test_gradient_centered_with_normalize(self):

        for ndim in [1, 2, 3]:
            dims = self.random_state.randint(50, 101, size=ndim)
            arr = self.random_state.randn(*dims)

            grads = mg.gradient_centered(arr, normalize=True,
                                         return_gradient_magnitude=False)

            gmag_normalize = np.linalg.norm(grads, axis=0)
            ones = np.ones_like(gmag_normalize)

            gmag_error = np.abs(gmag_normalize - ones).mean()

            self.assertLessEqual(gmag_error, 1e-8)

