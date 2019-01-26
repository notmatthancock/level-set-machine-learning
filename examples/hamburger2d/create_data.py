import numpy as np

from sklearn.ensemble import RandomForestRegressor

from level_set_machine_learning.data.dim2 import hamburger
from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.feature.provided.image import (
    ImageSample, ImageEdgeSample)
from level_set_machine_learning.initializer.provided.threshold import (
    ThresholdInitializer)

# Create the random number generator.
seed = 1234
rs = np.random.RandomState(seed)

n_samples = 50
img_dim = 51

# Create a dataset with 50 samples. The images are square, size 50px.
# rad, cb, cthick all set the ranges of the random values to generate
# the parameters of the images.
imgs, segs = hamburger.make_dataset(
        N=n_samples, n=img_dim, rad=[10, 16], cb=[1, 10], cthick=[1, 5],
        verbose=False)

# Set the path where the hdf5 file will be saved.
#h5file = './dataset.h5'

# Convert the arrays to an hdf5 file with correct schema for the SLLS routines.

features = [
    ImageSample(sigma=1, ndim=2),
    ImageEdgeSample(sigma=2, ndim=2)
]

lsml = LevelSetMachineLearning(
    features=features,
    initializer=ThresholdInitializer(sigma=1)
)

lsml.fit('dataset.h5', imgs=imgs, segs=segs,
         regression_model_class=RandomForestRegressor,
         regression_model_kwargs={}, max_iters=1)
