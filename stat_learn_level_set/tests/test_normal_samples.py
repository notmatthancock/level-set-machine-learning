import unittest

import numpy as np
import skfmm
from skimage.morphology import ball

from stat_learn_level_set.feature_maps.dim3.utils import normal_samples as ns
from stat_learn_level_set.utils import masked_grad as mg

#class TestNormalSamplesBall(unittest.TestCase):
#    def test_normal_samples_ball(self):
B = np.pad(ball(15), 20, 'constant')
B = 2*B.astype(np.float) - 1

dx = np.r_[3,2,1.]
#dx = np.ones(3)
nsamples = 10

dist = skfmm.distance(B[::3,::2,:], dx=dx)
#dist = skfmm.distance(B, dx=dx)
mask = ~skfmm.distance(dist, narrow=2, dx=dx).mask
ni,nj,nk = mg.gradient_centered(dist, mask=mask, dx=dx, normalize=True, return_gmag=False)
gmag = np.sqrt(ni**2 + nj**2 + nk**2)

com = np.r_[35,35,35.]

samples = np.zeros(dist.shape + (nsamples,2))

ns.get_samples(dist, ni, nj, nk, dx[0], dx[1], dx[2], com, samples, 
               mask=mask, nsamples=nsamples)


#if __name__ == '__main__':
#    unittest.main()
