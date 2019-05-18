import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickWriter

from level_set_machine_learning import LevelSetMachineLearning
from level_set_machine_learning.visualize import plot_iso_contours


# Load the model and an example
model = LevelSetMachineLearning.load('./LSML-model.pkl')
example = model.testing_data[13]


# Set up plotting for the movie frames
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.axis('off')
ax.imshow(example.img, cmap=plt.cm.gray, interpolation='bilinear')
lines = []
plot_iso_contours(ax, example.seg.astype(float), value=0.5, c='r')


# Set up the movie writer and grab the frame at initialization
writer = ImageMagickWriter(fps=7)
writer.setup(fig, 'seg-evolution.gif', 100)
writer.grab_frame()


# Define the callback function to be used during segmentation evolution
def update_movie(i, u):

    if i % 10 != 0:
        return

    for line in lines:
        line.remove()
    lines.clear()

    lines.extend(
        plot_iso_contours(ax, u, value=0, c='b')
    )
    ax.set_title("Iteration: {:d}".format(i))

    writer.grab_frame()


us = model.segment(img=example.img, on_iterate=[update_movie], verbose=True)

# Close out the movie writing
writer.finish()


