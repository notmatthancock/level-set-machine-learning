import numpy as np

import matplotlib.pyplot as plt

from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.visualize import interactive2d

from level_set_machine_learning.util.on_iterate import plot_contours, collect_scores


model = LevelSetMachineLearning.load('./LSML-model.pkl')
r = np.random.randint(len(model.testing_data))
print(r)
example = model.testing_data[r]

scores = []
score_collector = collect_scores(example.seg, scores, model.scorer)

plt.imshow(example.img, cmap=plt.cm.gray)

contour_plotter = plot_contours()

us = model.segment(img=example.img, on_iterate=[score_collector])

interactive2d(u=us, img=example.img, seg=example.seg, scores=scores)
