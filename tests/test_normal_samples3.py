import unittest

import numpy as np
import skfmm
from skimage.morphology import disk, ball

from level_set_learn.feature_maps.dim3.utils import normal_samples as ns
from level_set_learn.utils import masked_grad as mg

img = np.indices((71,71), dtype=np.float)[0] - 35
img = np.repeat(img[:,:,np.newaxis], 3, axis=2)

img1 = img.copy()
img2 = img[::3,::2,:].copy()

dx1 = np.ones(3, dtype=np.float)
dx2 = np.r_[3, 2, 1].astype(np.float)
nsamples = 10

mask1 = ~skfmm.distance(img1, narrow=3.0, dx=dx1).mask
mask2 = ~skfmm.distance(img2, narrow=3.0, dx=dx2).mask

#mask1 = np.ones(dist1.shape, dtype=np.bool)
#mask2 = np.ones(dist2.shape, dtype=np.bool)
#mask1 = np.repeat(mask1[:,:,np.newaxis], 3, axis=2)
#mask2 = np.repeat(mask2[:,:,np.newaxis], 3, axis=2)

grad1 = (np.ones(img1.shape),
         np.zeros(img1.shape),
         np.zeros(img1.shape))

grad2 = (np.ones(img2.shape),
         np.zeros(img2.shape),
         np.zeros(img2.shape))

inds1 = np.indices(img1.shape, dtype=np.float)
inds2 = np.indices(img2.shape, dtype=np.float)
com1 = [((img1 > 0)*i).sum() / (img1 > 0).sum() for i in inds1]
com2 = [((img2 > 0)*i).sum() / (img2 > 0).sum() for i in inds2]

#print com1, com2
#assert 0

samples1 = np.zeros(img1.shape + (nsamples,2))
samples2 = np.zeros(img2.shape + (nsamples,2))

ns.get_samples(img1, grad1[0], grad1[1], grad1[2], dx1[0], dx1[1], dx1[2], 
               com1, samples1, mask=mask1, nsamples=nsamples)

ns.get_samples(img2, grad2[0], grad2[1], grad2[2], dx2[0], dx2[1], dx2[2], 
               com2, samples2, mask=mask2, nsamples=nsamples)


#if __name__ == '__main__':
#    unittest.main()
