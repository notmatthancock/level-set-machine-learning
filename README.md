![Travis](https://travis-ci.com/notmatthancock/level-set-machine-learning.svg?token=vyEDTSsnnxok9zbF6h68&branch=master)


## Level set machine learning

Level set image segmentation with velocity field from
machine learning methods

![Hamburger](https://raw.githubusercontent.com/notmatthancock/level-set-machine-learning/master/examples/hamburger2d/seg-evolution.gif)

### Description of the method

In the standard level set segmentation
approach, the boundary of the approximating segmentation is given
by the zero level set of some scalar field u (also called the 
"level set function"), and the movement of zero level set of u
is prescribed to move in the normal direction with velocity ν, which
usually takes into account underlying image information.

This prescribed movement is summarized in the following PDE [1]:

u<sub>t</sub> = ν ||Du||<sub>2</sub>

In our machine learning extension of this method, the velocity
field ν is learned from data via machine learning regression methods,
rather than fixing the velocity function a priori based on expected image 
appearance (e.g., assuming object boundaries being defined by strong
image gradients).

The method accepts a feature set, wherein particular problem domain
characteristics and heuristics can be encoded, and the mapping from this
feature set to local level set velocities is delegated to machine learning
based regression techniques.

> [1]: Malladi, Ravi, James A. Sethian, and Baba C. Vemuri. "Shape modeling with front propagation: A level set approach." IEEE transactions on pattern analysis and machine intelligence 17.2 (1995): 158-175.
