from lsml import LevelSetMachineLearning
from lsml.visualize import interactive2d


# Load the model and grab an example from the testing set
model = LevelSetMachineLearning.load('./LSML-model.pkl')
example = model.testing_data[0]

# Segment the example image
us = model.segment(img=example.img)

# View the example interactively
interactive2d(u=us, img=example.img, seg=example.seg)
