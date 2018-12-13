import unittest

import numpy as np
from level_set_learn.neural_network import neural_network as nn


class TestStepRecovery(unittest.TestCase):
    def test_step_recovery(self):
        rs = np.random.RandomState(1234)

        ninput = 10
        nhidden = 1024
        nnet = nn.neural_network(ninput, nhidden, rs=rs)

        xtr = rs.randn(100,10)
        ytr = rs.randn(100)

        xva = rs.randn(100,10)
        yva = rs.randn(100)

        nnet.gradient_descent(xtr, ytr, xva, yva, iters=100, step=100.0)

if __name__ == '__main__':
    unittest.main()
