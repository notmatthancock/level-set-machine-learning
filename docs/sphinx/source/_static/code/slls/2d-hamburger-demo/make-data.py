import numpy as np  # from RandomState

from level_set_machine_learning.data.synth.dim2 import hamburger
from level_set_machine_learning.data import tohdf5

# Create the random number generator.
seed = 1234
rs = np.random.RandomState(seed)

n_samples = 50
img_dim = 51

# Create a dataset with 50 samples. The images are square, size 50px.
# rad, cb, cthick all set the ranges of the random values to generate
# the parameters of the images.
imgs,segs = hamburger.make_dataset(N=n_samples, n=img_dim,
                                   rad=[10,16], cb=[1,10], cthick=[1,5])

# Set the path where the hdf5 file will be saved.
h5file = './dataset.h5'

# Convert the arrays to an hdf5 file with correct schema for the SLLS routines.
tohdf5.convert(imgs, segs, h5file)
