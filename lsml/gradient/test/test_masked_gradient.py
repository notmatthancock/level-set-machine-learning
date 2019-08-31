import unittest

import numpy as np

from lsml.gradient import masked_gradient as mg


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

    def manual_gmag_os(self, arr, nu, dx):
        """ Manually compute the osher sethian gradient magnitude
        """
        di = np.diff(arr, axis=0) / dx[0]
        dj = np.diff(arr, axis=1) / dx[1]

        fwdi = np.vstack([di, di[-1]])
        bcki = np.vstack([di[0], di])

        fwdj = np.c_[dj, dj[:, -1]]
        bckj = np.c_[dj[:, 0], dj]

        plus = np.sqrt(np.maximum(bcki, 0)**2 + np.minimum(fwdi, 0)**2 +
                       np.maximum(bckj, 0)**2 + np.minimum(fwdj, 0)**2)

        minus = np.sqrt(np.minimum(bcki, 0)**2 + np.maximum(fwdi, 0)**2 +
                        np.minimum(bckj, 0)**2 + np.maximum(fwdj, 0)**2)

        gmag = np.zeros_like(arr)
        gmag[nu > 0] = minus[nu > 0]
        gmag[nu < 0] = plus[nu < 0]

        return gmag

    def test_gradient_magnitude_osher_sethian2d(self):

        arr = self.random_state.randn(141, 112)
        nu = self.random_state.randn(141, 112)
        dx = self.random_state.rand(2)

        gmag = mg.gradient_magnitude_osher_sethian(arr, nu, dx=dx)
        gmag_true = self.manual_gmag_os(arr, nu, dx)

        gmag_error = np.abs(gmag - gmag_true).mean()

        self.assertLessEqual(gmag_error, 1e-8)
