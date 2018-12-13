import unittest

import numpy as np
import skfmm
from skimage.morphology import disk, ball

from level_set_learn.feature_maps.dim3.utils import normal_samples as ns
from level_set_learn.utils import masked_grad as mg

#class TestNormalSamplesBall(unittest.TestCase):
#    def test_normal_samples_ball(self):
rad = 15
pad = 20
B = np.pad(ball(rad), pad, 'constant')
B = 2*B.astype(np.float) - 1


dx1 = np.r_[3,2,1.]
dx2 = np.ones(3)
nsamples = 10

dist1 = skfmm.distance(B[::3,::2,:], dx=dx1)
dist2 = skfmm.distance(B           , dx=dx2)

mask1 = ~skfmm.distance(dist1, narrow=3.0, dx=dx1).mask
mask2 = ~skfmm.distance(dist2, narrow=3.0, dx=dx2).mask

#dist1 = np.repeat(dist1[:,:,np.newaxis], 3, axis=2)
#dist2 = np.repeat(dist2[:,:,np.newaxis], 3, axis=2)
#
#mask1 = np.repeat(mask1[:,:,np.newaxis], 3, axis=2)
#mask2 = np.repeat(mask2[:,:,np.newaxis], 3, axis=2)

grad1, gmag1 = mg.gradient_centered(dist1, mask=mask1, dx=dx1, 
                                    normalize=True, return_gmag=True)
grad2, gmag2 = mg.gradient_centered(dist2, mask=mask2, dx=dx2, 
                                    normalize=True, return_gmag=True)

#raise Exception()

com1 = 0.5*np.array(dist1.shape)
com2 = np.r_[35,35,35.]

inds = np.indices(B.shape, dtype=np.float)
com2 = [((dist2 > 0)*i).sum() / (dist2 > 0).sum() for i in inds]
com1 = [((dist1 > 0)*i).sum() / (dist1 > 0).sum() for i in inds[:,::3,::2,:]]

print com1, com2

samples1 = np.zeros(dist1.shape + (nsamples,2))
samples2 = np.zeros(dist2.shape + (nsamples,2))

ns.get_samples(dist1, grad1[0], grad1[1], grad1[2], dx1[0], dx1[1], dx1[2],
               com1, samples1, mask=mask1, nsamples=nsamples)

ns.get_samples(dist2, grad2[0], grad2[1], grad2[2], dx2[0], dx2[1], dx2[2], 
               com2, samples2, mask=mask2, nsamples=nsamples)


#if __name__ == '__main__':
#    unittest.main()
