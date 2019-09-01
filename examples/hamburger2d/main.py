import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lsml.data.dim2 import hamburger
from lsml import LevelSetMachineLearning
from lsml.feature import get_basic_image_features, get_basic_shape_features
from lsml.initializer import RandomBallInitializer


random_state = np.random.RandomState(1234)


# Create a toy dataset ########################################################

n_samples = 200
imgs, segs = hamburger.make_dataset(
    N=n_samples, random_state=random_state)

# Set up the model and fit it #################################################

lsml = LevelSetMachineLearning(
    features=get_basic_image_features() + get_basic_shape_features(),
    initializer=RandomBallInitializer(random_state=random_state)
)

lsml.fit(
    'dataset.h5', imgs=imgs, segs=segs,
    max_iters=500, random_state=random_state,

    # We use a sklearn Pipeline as the regression model which is
    # simply a standard scaler followed by linear regression
    regression_model_class=Pipeline,
    regression_model_kwargs=dict(
        steps=[
            ('standardscaler', StandardScaler()),
            ('linearregressor', LinearRegression()),
        ],
     )
)
