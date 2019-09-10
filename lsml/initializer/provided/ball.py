import numpy

from lsml.initializer.initializer_base import InitializerBase


class BallInitializer(InitializerBase):
    """ Initialize the zero level set to a ball of fixed radius """

    def __init__(self, radius=10, location=None):
        self.radius = radius
        self.location = location

    def initialize(self, img, dx, seed):

        if self.location is not None and len(self.location) != img.ndim:
            msg = '`location` is len {} but should be {}'
            raise ValueError(msg.format(len(self.location), img.ndim))

        if self.location is None:
            location = 0.5 * numpy.array(img.shape)
        else:
            location = self.location

        # Used for broadcasting ...
        slices = (slice(None),) + tuple(None for _ in range(img.ndim))

        indices = numpy.indices(img.shape, dtype=float)
        indices *= dx[slices]
        indices -= (location * dx)[slices]

        return (self.radius - numpy.sqrt((indices**2).sum(axis=0))) > 0


class RandomBallInitializer(InitializerBase):
    """ Initialize the zero level set to a circle/sphere/hyper-sphere
    with random center and radius
    """

    def __init__(self, randomize_center=True, random_state=None):
        """ Initialize a RandomBallInitializer initialization object

        Parameters
        ----------
        random_state: numpy.random.RandomState, default None
            Supply for reproducible results

        randomize_center: bool
            If True, then location of the random ball is randomized

        """

        if random_state is None:
            random_state = numpy.random.RandomState()

        self.random_state = random_state
        self.randomize_center = randomize_center

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

        # Save the state to be reset later
        state = self.random_state.get_state()
        self.random_state.seed(seed_value)

        # Generate a random radius
        min_dim = min(dx * img.shape)
        radius = self.random_state.uniform(
            low=0.20*min_dim, high=0.25*min_dim)

        indices = [numpy.arange(img.shape[i], dtype=numpy.float)*dx[i]
                   for i in range(img.ndim)]

        # Select the center point uniformly at random.
        # Expected center is at the center of image, but could
        # be terribly far away in general.
        if self.randomize_center:
            center = []
            for i in range(img.ndim):
                while True:
                    center_coord = self.random_state.choice(indices[i])
                    if (center_coord-radius > indices[i][0] and
                            center_coord+radius <= indices[i][-1]):
                        center.append(center_coord)
                        break
            center = numpy.array(center)
        else:
            center = 0.5 * numpy.array(img.shape, dtype=numpy.float)

        indices = numpy.indices(img.shape, dtype=numpy.float)
        shape = dx.shape + tuple(numpy.ones(img.ndim, dtype=int))
        indices *= dx.reshape(shape)
        indices -= center.reshape(shape)
        indices **= 2

        init_mask = indices.sum(axis=0)**0.5 <= radius

        # Reset the random state state
        self.random_state.set_state(state)

        return init_mask
