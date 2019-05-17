import numpy as np

from sklearn.linear_model import LinearRegression

from level_set_machine_learning.data.dim2 import gestalt_triangle
from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.feature.provided.image import (
    ImageSample, ImageEdgeSample,
    InteriorImageAverage, InteriorImageVariation)
from level_set_machine_learning.feature.provided.shape import (
    BoundarySize, DistanceToCenterOfMass, IsoperimetricRatio, Moment, Size)
from level_set_machine_learning.initializer.initializer_base import (
    InitializerBase)

from corner_feature import CornerFeature


class SimpleInit(InitializerBase):
    def __init__(self, radius=10):
        self.radius = radius

    def initialize(self, img, dx, seed):
        from skimage.morphology import disk
        return np.pad(
            disk(self.radius),
            img.shape[0]//2-self.radius,
            'constant'
        ).astype(np.bool)


random_state = np.random.RandomState(1234)

############################################################
# Create the dataset
############################################################

n_samples = 200
imgs, segs = gestalt_triangle.make_dataset(
    N=n_samples, rs=random_state)

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
    CornerFeature(sigma=0),
    #CornerFeature(sigma=2),
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
    initializer=SimpleInit()
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import subsample_rf as srf


lsml.fit('dataset.h5', imgs=imgs, segs=segs,
         regression_model_class=Pipeline,
         regression_model_kwargs=dict(
             steps=[
                 ('standardscaler', StandardScaler()),
                 ('svr', srf.SubsampleRF(n_estimators=100, random_state=random_state)),
             ],
         ),
         max_iters=50, random_state=random_state)
