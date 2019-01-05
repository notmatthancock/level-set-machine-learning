import numpy

from level_set_machine_learning.initialize.initialize_base import (
    InitializeBase)


class RandomBallInitialize(InitializeBase):
    """ Initialize the zero level set to a circle/sphere/hyper-sphere
    with random center and radius
    """

    def __init__(self, random_state=None):
        """ Initialize a RandomBallInitialize initialization object

        Parameters
        ----------
        random_state: numpy.random.RandomState, default None
            Supply for reproducible results

        """

        if random_state is None:
            self.random_state = numpy.random.RandomState()
        elif isinstance(random_state, numpy.random.RandomState):
            self.random_state = random_state

    def _get_seed_value_from_image(self, img):
        """ Uses the first integer 4 values after the decimal point of the
        first image value as the seed
        """
        img_val = img.ravel()[0]
        img_str = "{:.4f}".format(img_val)

        _, decimal_str = img_str.split(".")
        seed_val = int(decimal_str)

        return seed_val

    def initialize(self, img, dx, seed):

        # Seed the random state from the image so that the same "random"
        # initialization is given for identical image inputs
        seed_value = self._get_seed_value_from_image(img)
        self.random_state.seed(seed_value)

        # Generate a random radius
        radius = (5 * dx.min() +
                  self.random_state.rand() *
                  min(dx * img.shape) * 0.25)

        indices = [numpy.arange(img.shape[i], dtype=numpy.float)*dx[i]
                   for i in range(img.ndim)]
        
        # Select the center point uniformly at random.
        # Expected center is at the center of image, but could
        # be terribly far away in general.
        center = []
        for i in range(img.ndim):
            while True:
                center_coord = self.random_state.choice(indices[i])
                if (center_coord-radius > indices[i][0] and
                        center_coord+radius <= indices[i][-1]):
                    center.append(center_coord)
                    break
        center = numpy.array(center)

        indices = numpy.indices(img.shape, dtype=numpy.float)
        shape = dx.shape + tuple(numpy.ones(img.ndim, dtype=int))
        indices *= dx.reshape(shape)
        indices -= center.reshape(shape)
        indices **= 2
        
        init_mask = indices.sum(axis=0)**0.5 <= radius

        return init_mask
