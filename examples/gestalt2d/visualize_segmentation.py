import numpy as np

from lsml import LevelSetMachineLearning
from lsml.visualize import interactive2d
from lsml.initializer.initializer_base import InitializerBase

from sklearn.model_selection import GridSearchCV

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


model = LevelSetMachineLearning.load('./LSML-model.pkl')
r = 116#np.random.randint(len(model.training_data)); print(r)
example = model.training_data[r]

us, scores = model.segment(
    img=example.img, seg=example.seg, return_scores=True, verbose=True,
    iterate_until_validation_max=False)

interactive2d(u=us, img=example.img, seg=example.seg, scores=scores)
