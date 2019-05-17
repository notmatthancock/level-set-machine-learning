import numpy as np

from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.visualize import interactive2d


model = LevelSetMachineLearning.load('./LSML-model.pkl')
model.initializer.randomize_center = True
example = model.testing_data[13]

us, scores = model.segment(
    img=example.img, seg=example.seg, return_scores=True, verbose=False,
    iterate_until_validation_max=False)

interactive2d(u=us, img=example.img, seg=example.seg, scores=scores)
