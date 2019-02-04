import numpy as np

from sklearn.ensemble import RandomForestRegressor

from level_set_machine_learning.data.dim2 import hamburger
from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.feature.provided.image import (
    ImageSample, ImageEdgeSample)
from level_set_machine_learning.feature.provided.shape import (
    BoundarySize, IsoperimetricRatio, Moment, Size)
#from level_set_machine_learning.initializer.provided.random_ball import (
#    RandomBallInitializer)
from level_set_machine_learning.initializer.provided import ThresholdInitializer

# Create a dataset with 50 samples with randomly generated "hamburger" images
n_samples = 50
imgs, segs = hamburger.make_dataset(N=n_samples)

# Set up the features to be used
features = [
    # Image features
    ImageSample(sigma=0, ndim=2),
    ImageSample(sigma=2, ndim=2),
    ImageEdgeSample(sigma=0, ndim=2),
    ImageEdgeSample(sigma=2, ndim=2),

    # Shape features
    Size(ndim=2),
    BoundarySize(ndim=2),
    IsoperimetricRatio(ndim=2),
    Moment(ndim=2, axis=0, order=1),
    Moment(ndim=2, axis=1, order=1),
    Moment(ndim=2, axis=0, order=2),
    Moment(ndim=2, axis=1, order=2),
]

# Instantiate the model with features and initializer
lsml = LevelSetMachineLearning(
    features=features,
    #initializer=RandomBallInitializer()
    initializer=ThresholdInitializer(sigma=3)
)

# Fit the model
random_state = np.random.RandomState(1234)
lsml.fit('dataset.h5', imgs=imgs, segs=segs,
         regression_model_class=RandomForestRegressor,
         regression_model_kwargs={'n_estimators': 10,
                                  'random_state': random_state},
         max_iters=150, random_state=random_state)
