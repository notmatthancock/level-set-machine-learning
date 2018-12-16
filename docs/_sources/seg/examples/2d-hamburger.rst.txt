Segment 2d images
-----------------

In this example, we'll illustrate the SLLS programs on some simple 
two-dimensional images like the one shown below (which we are calling
'hamburger images' since it looks like hamburger buns).

.. literalinclude:: /_static/code/slls/2d-hamburger-demo/sample-image.py
    :language: python


.. raw:: html

    <img src="/_static/images/2d-hamburger-example.png">

We have used the `hamburger` module from 
the `LevelSetMachineLearning.utils.dim2.hamburger` module to create
the above image, using a cut oriented at :math:`\pi/4` radians through the
circle, using the `hamburger.make` function.

The SLLS programs include a number of utilities for creating synthetic 
datasets for testing purposes that can be found in the
`utils.data.synth` submodule, which is further divided into the submodules
`dim2` and `dim3`.

We can make a dataset of randomly generated images for image segmentation 
testing purposes using `hamburger.make_dataset()`. This will randomize 
the various parameters (e.g., radius, cut angle, cut thickness, noise 
strength, etc.) for each image. The result is two numpy `numpy.ndarray`'s:
the images and the corresponding segmentations. In the following,
we'll using a `numpy.random.RandomState` object to seed the random number
generator for reproducible results in making the dataset:

After generating the image and segmentation pairs as `numpy` arrays, we
store them in an `hdf5` file that has the schema expected by the SLLS
routines. This is accomplished using the 
`LevelSetMachineLearning.utils.tohdf5` utility.

.. literalinclude:: /_static/code/slls/2d-hamburger-demo/make-data.py
    :language: python

The dataset is now stored and ready for use. Let's now run the SLLS
training routine. To do so, we first create a `LevelSetMachineLearning`
object, and provide some specifications. Next, we further specificy some
parameters of the fitting process as well as the neural network fitting 
process (which happens at each iteration of the global fitting process).

An example of the fitting routine is given in the following block of code:

.. literalinclude:: /_static/code/slls/2d-hamburger-demo/main.py
    :language: python

In the previous code block, we start by specifying the feature map
instance to be an instance of the `simple feature map </slls/api/feature_maps/dim2/simple_feature_map.html>`_
which is provided by the `LevelSetMachineLearning` module. This feature map computes simple
local image and image gradient features, as well as some global image and shape features. 
The features are computed at multiple scales, namely at the sigma values provided at
initialized. The `simple feature map </slls/api/feature_maps/dim2/simple_feature_map.html>`_
derives from a `base class </slls/api/feature_maps/base.html>`_, and all other feature maps that are developed and used to 
plug into the slls module should also derive from the same base class.

Next, we specify the initialization function to be an instance of the 
`random </slls/api/init_funcs/random.html>`_, which initializes the zero level 
set to a circle of random radius and random location. Ideally, a more robust 
initialization function should be developed and used. Any initialization function
should derive from the `initialization base class </slls/api/init_funcs/base.html>`_.
The `LevelSetMachineLearning` module also provides another simple initialization routine that uses
`thresholding </slls/api/init_funcs/threshold.html>`_.
