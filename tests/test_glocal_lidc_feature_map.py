import h5py
import numpy as np

import pylidc.utils as pu

from slls.feature_maps.dim3 import \
        lidc_glocal_feature_map

with h5py.File('/home/matt/LIDC/3d/no-interp/dataset.h5', 'r') as hf:
    img = hf['0/img'][...]
    seg = hf['0/seg'][...]
    dist = hf['0/dist'][...]
    u = 2.0*(dist > 0) -1.
    #    mask = np.abs(dist) < 3.0
    mask = np.ones(dist.shape, dtype=np.bool)
    dx = hf['0'].attrs['dx']

fmap = lidc_glocal_feature_map.lidc_glocal_feature_map(sigmas=[0])

F = fmap(u=u, img=img, dist=dist, mask=mask, dx=dx)
