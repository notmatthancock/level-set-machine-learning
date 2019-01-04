import skfmm
import numpy as np

from scipy.ndimage import gaussian_filter1d as gf1d
from skimage.measure import label

from level_set_machine_learning.initialization import InitializationBase
from .util import radii_from_mask as rfm


class RayTrim(InitializationBase):
    """
    Local thresholding followed by radius trimming.

    Example::

        TODO
        
    """
    def __init__(self, sigma=4, adjust_sigma=True, pr=60, ntpr=1000, 
                 ball_small=True, min_vol=40.0, alert_small=True, rs=None):
        """
        sigma: float or list of floats, len=3
            The list of Gaussian blur factors for each axis. If a single
            float is provided, then the same provided sigma value is used
            along each axis.

        adjust_sigma: bool, default=True
            Adjust the provided `sigma` values by taking into account the
            provided `dx` values when `__call__` is called.

        pr: float, default=60
            The radius percentile in range [0, 100]. Radii outside the 
            computed `pr` percentile are clipped from the initial segmentation.

        ntpr: int, default=1000
            The number of theta and phi angles to sample radii over.

        ball_small: bool, default=True
            If the volume of the initialization is less than `min_vol`
            then the initialization is simply set to a ball of volume
            `min_vol`.

        min_vol: float, default=40.0
            See `ball_small` parameter.

        alert_small: bool, default=True
            Print message when initialization is replaced with the small ball.

        random_state: numpy.random.RandomState, default=None
            Provide for reproducibility.
        """
        if isinstance(sigma, int) or isinstance(sigma, int):
            self.sigma = np.array([sigma, sigma, sigma], dtype=np.float)
        elif np.iterable(sigma):
            if len(sigma) != 3:
                raise ValueError("`sigma` should be length 3.")
            self.sigma = np.array(sigma, dtype=np.float)
        else:
            raise TypeError("`sigma` is not correct type.")

        self.adjust_sigma = adjust_sigma

        if pr <= 0 or pr > 100:
            raise ValueError("`pr` should be in range (0, 100]")
        self.pr = pr

        self.ntpr = ntpr
        self.rs = np.random.RandomState() if rs is None else rs
        self.ball_small = ball_small
        self.min_vol = min_vol
        self.alert_small = alert_small

        # ... End input variable checking.

        self.X = self.rs.randn(ntpr, 3)
        self.X /= np.linalg.norm(self.X, axis=1).reshape(-1, 1)

        # This variable will be re-used in the `__call__` function.
        self.radii = np.zeros(ntpr)

        # Compute thetas.
        self.thetas = np.arctan2(self.X[:, 1], self.X[:, 0])

        # Move thetas from range [-pi, pi] to [0, 2*pi].
        self.thetas[self.thetas < 0] = self.thetas[self.thetas < 0] + 2*np.pi

        # Compute phis.
        self.phis = np.arccos(self.X[:, 2])

    def __call__(self, img, band, dx=None, seed=None, only_seg=False):
        """
        `seed` should not account for `dx`, i.e., it should be provided
        in "index" coordinates (although fractional coordinates are allowed).
        """
        dx = np.ones(3) if dx is None else np.array(dx)

        if seed is None:
            # Make seed the center voxel if not provided.
            seed = 0.5*np.array(img.shape)
        elif len(seed) != 3:
            raise ValueError("`seed` should be length 3.")
        seed = np.array(seed, dtype=np.float)
        
        # Some local variables.
        grids = [np.arange(img.shape[i], dtype=np.float)*dx[i]
                  for i in range(3)]
        ii, jj, kk = np.meshgrid(*grids, indexing='ij')
        dist = np.sqrt((ii-seed[0]*dx[0])**2 + 
                       (jj-seed[1]*dx[1])**2 + 
                       (kk-seed[2]*dx[2])**2)

        # Local thresholding
        G = img.copy()
        for i in range(3):
            G = gf1d(G, self.sigma[i]/dx[i], axis=i)
        B = (img >= G)

        inds = np.array(np.where(B))
        q = ((inds[0] - seed[0])**2 +\
             (inds[1] - seed[1])**2 +\
             (inds[2] - seed[2])**2).argmin()
        seed = np.array([i[q] for i in inds])

        # Compute radii and determine radius threshold.
        rfm.radii_from_mask(thetas=self.thetas, phis=self.phis, 
                            seed=seed*dx, mask=B, radii=self.radii,
                            di=dx[0], dj=dx[1], dk=dx[2])
        rad_thresh = np.percentile(self.radii, self.pr)

        # Set voxels outside of radius threshold to zero.
        B[dist > rad_thresh] = False

        # Remove non-connected components that arise from the radius clip.
        L, num = label(B, return_num=True)
        if num > 1: B = (L == L[tuple(seed.round().astype(np.int))])

        # If B is too small, initialize to a ball of volume ~ `min_vol`.
        if self.ball_small and B.sum()*np.prod(dx) < self.min_vol:
            B[...] = False
            rad = (self.min_vol / (4/3.) / np.pi)**(1./3)
            B[dist <= rad] = True

            # If the resolution doesn't permit a ball of radius `min_vol`
            # then set the seed pixel to true
            if B.sum() == 0:
                B[seed[0], seed[1], seed[2]] = True

            if self.alert_small:
                print("Volume less than `min_vol`. Small ball is used.")

        if only_seg: return B
        
        u0 = B.astype(np.float)
        u0 *= 2; u0 -= 1

        dist = skfmm.distance(u0, narrow=band, dx=dx)

        if hasattr(dist, 'mask'):
            mask = ~dist.mask
            dist = dist.data
        else:
            mask = np.ones(img.shape, dtype=np.bool)

        return u0, dist, mask
