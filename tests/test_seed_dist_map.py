import shutil

import numpy as np
import h5py
import skfmm

from stat_learn_level_set.utils import seed


ii,jj = np.indices((101, 75), dtype=np.float)

hf = h5py.File('/home/matt/LIDC/3d/no-interp/dataset.h5')
seeds = seed.sample_from_distance_map(hf, seeds_per_image = 10)
