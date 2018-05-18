from stat_learn_level_set.init_funcs import init_func_base
import numpy as np
import skfmm

class random(init_func_base):
    """
    Initialize the zero level set to a sphere
    of random location and radius.

    This initialization function works for all dimensions, so
    the intialization is a random circle in 2d, sphere in 3d, etc.

    If the `reproducible` flag is set to True,
    then the same "random" initialization is produced
    when the same image is given as input.

    Example::
        
        >>> from stat_learn_level_set.init_funcs import random as init_random

        >>> ifnc = init_random.random(reproducible=True)
        >>> img = np.random.randn(46, 67, 81)

        >>> u0,dist,mask = ifnc(img, band=3.0)
    """
    def __init__(self, reproducible=True):
        self.rs = np.random.RandomState()
        self.reproducible = reproducible

    def __call__(self, img, band, dx=None):
        dx = np.ones(img.ndim) if dx is None else dx

        if self.reproducible:
            # Create the seed value from the image.
            s = str(img.flatten()[0])
            n = s.index(".")
            seed_val = int(s[n+1 : n+1+4])
            self.rs.seed(seed_val)

        # Select the radius.
        r = 5*dx.min() + self.rs.rand() * min(dx * img.shape) * 0.25

        inds = [np.arange(img.shape[i], dtype=np.float)*dx[i]
                 for i in range(img.ndim)]
        
        # Select the center point uniformly at random.
        # Expected center is at the center of image, but could
        # be terribly far away in general.
        center = []
        for i in range(img.ndim):
            while True:
                c = self.rs.choice(inds[i])
                if c-r > inds[i][0] and c+r <= inds[i][-1]:
                    center.append(c)
                    break
        center = np.array(center)

        inds = np.indices(img.shape, dtype=np.float)
        shape = dx.shape + tuple(np.ones(img.ndim, dtype=int))
        inds *= dx.reshape(shape)
        inds -= center.reshape(shape)
        inds **= 2
        
        u0 = (inds.sum(axis=0)**0.5 <= r).astype(np.float)
        u0 *= 2
        u0 -= 1

        dist = skfmm.distance(u0, narrow=band, dx=dx)

        if hasattr(dist, 'mask'):
            mask = ~dist.mask
            dist = dist.data
        else:
            mask = np.ones(img.shape, dtype=np.bool)

        return u0, dist, mask
