import unittest
import numpy as np

from stat_learn_level_set.utils import masked_grad as mg

class TestMaskedGrad(unittest.TestCase):
    def test_linear(self):
        tol = 1e-9
        rs = np.random.RandomState(3)
        N = 100

        for attempt in range(N):
            ii,jj,kk = np.indices(tuple(rs.randint(3, 71, size=3)),
                                  dtype=np.float)

            # Create a linear function to test.
            coefs = rs.randn(4)
            func  = lambda x,y,z: coefs[0]*x + \
                                  coefs[1]*y + \
                                  coefs[2]*z + \
                                  coefs[3]

            dx = rs.rand(3)*5
            vals = func(ii*dx[0], jj*dx[1], kk*dx[2])

            grad = mg.gradient_centered(vals, dx=dx, normalize=False,
                                        return_gmag=False)

            for i in range(3):
                self.assertLessEqual(np.abs(coefs[i]-grad[i]).mean(), tol)


if __name__ == '__main__':
    unittest.main()
