![Travis](https://travis-ci.com/notmatthancock/level-set-machine-learning.svg?token=vyEDTSsnnxok9zbF6h68&branch=master)


## Level set machine learning

Level set image segmentation with velocity field from
machine learning methods

### Approach

In the standard level set segmentation
approach, the boundary of the approximating segmentation is given
by the zero level set of some scalar field u (also called the 
"level set function"), and the movement of zero level set of u
is prescribed to move in the normal direction with velocity ν, which
usually takes into account underlying image information.

This prescribed movement is summarized in the following PDE:

u<sub>t</sub> = ν || Du ||

In the level set machine learning method, the velocity field ν
is learned from data via machine learning regression methods,
rather than fixed a priori based on heuristics about expected
underlying images.
