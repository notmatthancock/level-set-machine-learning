import numpy as np

from level_set_machine_learning.utils.visualize import interactive2d


model = np.load('./lsl_model.pkl')
key = np.random.choice(model._datasets['ts'])

with model._data_file() as df:
    img = df["{}/img".format(key)][...]
    seg = df["{}/seg".format(key)][...]

u, scores = model.segment(img=img, seg=seg, verbose=True)

interactive2d(u=u, img=img, seg=seg, scores=scores)

