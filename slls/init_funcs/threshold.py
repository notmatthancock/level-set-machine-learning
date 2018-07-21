import skfmm
import numpy as np
from slls.init_funcs import init_func_base
from scipy.ndimage import gaussian_filter as gf
from skimage.filters.thresholding import threshold_otsu as otsu

class threshold(init_func_base):
    """
    Use otsu thresholding to initialize the level set.

    This initialization function is independent of dimension.

    Example::
        
        >>> from slls.init_funcs import threshold as th

        >>> ifnc = th.threshold(sigma=2)
        >>> img = np.random.randn(46, 67, 81)

        >>> u0,dist,mask = ifnc(img, band=3.0)
    """
    def __init__(self, sigma=1.234):
        assert sigma >= 0
        self.sigma = sigma

    def __call__(self, img, band, dx=None, seed=None):
        dx = np.ones(img.ndim) if dx is None else dx

        blur = img if self.sigma == 0 else gf(img, self.sigma)
        t = otsu(blur)
        
        u0 = (blur >= t).astype(np.float)
        u0 *= 2
        u0 -= 1

        dist = skfmm.distance(u0, narrow=band, dx=dx)

        if hasattr(dist, 'mask'):
            mask = ~dist.mask
            dist = dist.data
        else:
            mask = np.ones(img.shape, dtype=np.bool)

        return u0, dist, mask
