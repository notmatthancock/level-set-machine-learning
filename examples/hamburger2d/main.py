import numpy as np

from sklearn.linear_model import LinearRegression

from level_set_machine_learning.data.dim2 import hamburger
from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.feature.provided.image import (
    ImageSample, ImageEdgeSample)
from level_set_machine_learning.feature.provided.shape import (
    BoundarySize, DistanceToCenterOfMass, IsoperimetricRatio, Moment, Size)
from level_set_machine_learning.initializer.provided.random_ball import (
    RandomBallInitializer)
from level_set_machine_learning.initializer.provided.threshold import (
    ThresholdInitializer)


random_state = np.random.RandomState(1234)

############################################################
# Create the dataset
############################################################

n_samples = 200
imgs, segs = hamburger.make_dataset(N=n_samples, random_state=random_state)

############################################################
# Set up features for feature map
############################################################

image_features = [
    ImageSample(sigma=0, ndim=2),
    ImageSample(sigma=2, ndim=2),
    ImageEdgeSample(sigma=0, ndim=2),
    ImageEdgeSample(sigma=2, ndim=2),
]

shape_features = [
    DistanceToCenterOfMass(ndim=2),
    Size(ndim=2),
    BoundarySize(ndim=2),
    IsoperimetricRatio(ndim=2),
    Moment(ndim=2, axis=0, order=1),
    Moment(ndim=2, axis=1, order=1),
    Moment(ndim=2, axis=0, order=2),
    Moment(ndim=2, axis=1, order=2),
]

############################################################
# Set up the model and fit it
############################################################

lsml = LevelSetMachineLearning(
    features=image_features + shape_features,
    initializer=RandomBallInitializer(random_state=random_state)
)

lsml.fit('dataset.h5', imgs=imgs, segs=segs,
         regression_model_class=LinearRegression,
         regression_model_kwargs={}, step=0.1,
         max_iters=300, random_state=random_state)
