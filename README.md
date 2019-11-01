![Travis](https://travis-ci.com/notmatthancock/level-set-machine-learning.svg?token=vyEDTSsnnxok9zbF6h68&branch=master)


## Level set machine learning

Level set image segmentation with velocity field from
machine learning methods

Example 1 | Example 2
:--------:|:---------:
![Hamburger2d](https://raw.githubusercontent.com/notmatthancock/level-set-machine-learning/master/img/hamburger2d.gif) | ![Gestalt2d](https://raw.githubusercontent.com/notmatthancock/level-set-machine-learning/master/img/gestalt2d.gif)

### Installation

Not on pip :( install locally:

```bash
git clone git@github.com:notmatthancock/level-set-machine-learning.git
cd level-set-machine-learning
pip install -e .
```

### Description of the method

In the standard level set segmentation
approach, the boundary of the approximating segmentation is given
by the zero level set of some scalar field u (also called the 
"level set function"), and the movement of zero level set of u
is prescribed to move in the normal direction with velocity ν, which
usually takes into account underlying image information.

This prescribed movement is summarized in the following PDE [1]:

u<sub>t</sub> = v ||Du||<sub>2</sub>

In this extension of the level set method method [2], the velocity
field v is learned from data via machine learning regression methods,
rather than fixing the velocity function a priori based on expected image 
appearance (e.g., assuming object boundaries being defined by strong
image gradients).

[1]: Malladi, Ravi, James A. Sethian, and Baba C. Vemuri. "Shape modeling with front propagation: A level set approach." IEEE transactions on pattern analysis and machine intelligence 17.2 (1995): 158-175.

[2]: Hancock, Matthew C., and Jerry F. Magnan. "Lung nodule segmentation via level set machine learning." arXiv preprint arXiv:1910.03191 (2019). https://arxiv.org/abs/1910.03191

### Examples

See `examples` directory for the method illustrated on some synthetic data.