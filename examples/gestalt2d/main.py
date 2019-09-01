import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lsml.data.dim2 import gestalt_triangle
from lsml import LevelSetMachineLearning
from lsml.feature import get_basic_image_features, get_basic_shape_features
from lsml.initializer import BallInitializer

from corner_feature import CornerFeature
from prev_iter_feature import PrevIterFeature


random_state = np.random.RandomState(1234)


# Create a toy dataset ########################################################

n_samples = 200
imgs, segs = gestalt_triangle.make_dataset(
    N=n_samples, random_state=random_state)

# Set up the model and fit it #################################################

corner_features = [CornerFeature(sigma=s) for s in range(4)]
prev_iter_features = [PrevIterFeature(sigma=s) for s in range(4)]

lsml = LevelSetMachineLearning(
    features=(
        get_basic_image_features(sigmas=[0, 1, 2, 3]) +
        get_basic_shape_features() +
        corner_features +
        prev_iter_features
    ),
    initializer=BallInitializer(radius=10)
)

lsml.fit(
    'dataset.h5', imgs=imgs, segs=segs,
    max_iters=10, random_state=random_state,

    # We use a sklearn Pipeline as the regression model which is
    # simply a standard scaler followed by random forest regression
    regression_model_class=Pipeline,
    regression_model_kwargs=dict(
        steps=[
            ('standardscaler', StandardScaler()),
            ('linearregressor', RandomForestRegressor(
                n_estimators=100, max_samples=300,
                random_state=random_state,
            )),
        ],
     )
)
