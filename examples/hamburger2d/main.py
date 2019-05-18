import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from level_set_machine_learning.data.dim2 import hamburger
from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.feature.provided.image import (
    ImageSample, ImageEdgeSample,
    InteriorImageAverage, InteriorImageVariation)
from level_set_machine_learning.feature.provided.shape import (
    BoundarySize, DistanceToCenterOfMass, IsoperimetricRatio, Moment, Size)
from level_set_machine_learning.initializer.provided.random_ball import (
    RandomBallInitializer)


random_state = np.random.RandomState(1234)


############################################################
# Create the dataset
############################################################

n_samples = 200
imgs, segs = hamburger.make_dataset(
    N=n_samples, random_state=random_state)

############################################################
# Set up features for feature map
############################################################

image_features = [
    ImageSample(sigma=0),
    ImageSample(sigma=2),
    ImageEdgeSample(sigma=0),
    ImageEdgeSample(sigma=2),
    InteriorImageAverage(sigma=0),
    InteriorImageAverage(sigma=2),
    InteriorImageVariation(sigma=0),
    InteriorImageVariation(sigma=2),
]

shape_features = [
    DistanceToCenterOfMass(),
    Size(),
    BoundarySize(),
    IsoperimetricRatio(),
    Moment(axis=0, order=1),
    Moment(axis=1, order=1),
    Moment(axis=0, order=2),
    Moment(axis=1, order=2),
]

############################################################
# Set up the model and fit it
############################################################

lsml = LevelSetMachineLearning(
    features=image_features + shape_features,
    initializer=RandomBallInitializer(random_state=random_state)
)

lsml.fit('dataset.h5', imgs=imgs, segs=segs,
         regression_model_class=Pipeline,
         regression_model_kwargs=dict(
             steps=[
                 ('standardscaler', StandardScaler()),
                 ('linearregressor', LinearRegression()),
             ],
         ),
         max_iters=500, random_state=random_state)
