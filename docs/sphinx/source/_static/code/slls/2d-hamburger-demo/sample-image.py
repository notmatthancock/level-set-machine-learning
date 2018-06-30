import matplotlib.pyplot as plt

from stat_learn_level_set.utils.data.synth.dim2 import hamburger

img, seg, params = hamburger.make(n=101, cut_theta=3.14159/4)

plt.imshow(img)
plt.show()
