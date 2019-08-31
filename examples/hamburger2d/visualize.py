import numpy as np

from lsml import LevelSetMachineLearning
from lsml.visualize import interactive2d


model = LevelSetMachineLearning.load('./LSML-model.pkl')
example = model.testing_data[13]

us, scores = model.segment(
    img=example.img, seg=example.seg, return_scores=True, verbose=False,
    iterate_until_validation_max=False)

interactive2d(u=us, img=example.img, seg=example.seg, scores=scores)
