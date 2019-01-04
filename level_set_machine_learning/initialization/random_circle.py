from level_set_machine_learning.initialization import InitializationBase
import numpy as np
import skfmm


class RandomCircle(InitializationBase):
    """
    Initialize the zero level set to a sphere
    of random location and radius.

    This initialization function works for all dimensions, so
    the initialization is a random circle in 2d, sphere in 3d, etc.

    If the `reproducible` flag is set to True,
    then the same "random" initialization is produced
    when the same image is given as input.

    Example::
        
        TODO
    """

    def __init__(self, random_state=None, reproducible=True):
        """ Initialize a RandomCircle initialization object

        Parameters
        ----------
        random_state: numpy.random.RandomState, default None
            Supply for reproducible results

        """

        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state
        self.reproducible = reproducible

    def __call__(self, img, band, dx=None, seed=None):

        dx = np.ones(img.ndim) if dx is None else dx

        if self.reproducible:
            # Create the seed value from the image.
            s = str(img.flatten()[0])
            n = s.index(".")
            seed_val = int(s[n+1 : n+1+4])
            self.random_state.seed(seed_val)

        # Select the radius.
        radius = (5 * dx.min() +
                  self.random_state.rand() * min(dx * img.shape) * 0.25)

        indices = [np.arange(img.shape[i], dtype=np.float)*dx[i]
                for i in range(img.ndim)]
        
        # Select the center point uniformly at random.
        # Expected center is at the center of image, but could
        # be terribly far away in general.
        center = []
        for i in range(img.ndim):
            while True:
                c = self.random_state.choice(indices[i])
                if c-radius > indices[i][0] and c+radius <= indices[i][-1]:
                    center.append(c)
                    break
        center = np.array(center)

        indices = np.indices(img.shape, dtype=np.float)
        shape = dx.shape + tuple(np.ones(img.ndim, dtype=int))
        indices *= dx.reshape(shape)
        indices -= center.reshape(shape)
        indices **= 2
        
        u0 = (indices.sum(axis=0)**0.5 <= radius).astype(np.float)
        u0 *= 2
        u0 -= 1

        dist = skfmm.distance(u0, narrow=band, dx=dx)

        if hasattr(dist, 'mask'):
            mask = ~dist.mask
            dist = dist.data
        else:
            mask = np.ones(img.shape, dtype=np.bool)

        return u0, dist, mask
