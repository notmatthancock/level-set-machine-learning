import skfmm
import numpy as np
from rbls.init_funcs import init_func_base
from skimage.filters.thresholding import threshold_otsu as otsu

class threshold(init_func_base):
    """
    Use otsu thresholding to initialize the level set.

    This initialization function is independent of dimension.

    Example::
        
        >>> from rbls.init_funcs import threshold as th

        >>> ifnc = th.threshold(reproducible=True)
        >>> img = np.random.randn(46, 67, 81)

        >>> u0,dist,mask = ifnc(img, band=3.0)
    """
    def __init__(self): pass

    def __call__(self, img, band, dx=None):
        dx = np.ones(img.ndim) if dx is None else dx

        t = otsu(img)
        
        u0 = (img >= t).astype(np.float)
        u0 *= 2
        u0 -= 1

        dist = skfmm.distance(u0, narrow=band, dx=dx)

        if hasattr(dist, 'mask'):
            mask = ~dist.mask
            dist = dist.data
        else:
            mask = np.ones(img.shape, dtype=np.bool)

        return u0, dist, mask
