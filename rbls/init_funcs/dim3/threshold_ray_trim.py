"""
See `init` function. Performs local thresholding and radial trimming.
"""
import sys
import numpy as np
import skfmm

from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d as gf1d
from skimage.measure import label
from rbls.cutils import radii_from_mask as rfm


rs = None
ntpr = None
X = None
radii = None
thetas = None
phis = None


# Make thetas/phis that are uniformly sampled on the sphere.
def make_tpr(n):
    global rs, ntpr, X, radii, thetas, phis

    rs = np.random.RandomState(123)
    ntpr = int(n/2)
    X = rs.randn(ntpr, 3)
    X /= np.linalg.norm(X, axis=1).reshape(ntpr,1)
    radii = np.zeros(2*ntpr)

    # Create theta and phi.
    thetas = np.arctan2(X[:,1], X[:,0])
    thetas[thetas < 0] = thetas[thetas < 0] + np.pi
    phis   = np.arccos(X[:,2])

    # Sort them by increasing theta.
    inds   = np.argsort(thetas)
    thetas = thetas[inds]
    phis   = phis[inds]

    ltp = thetas < np.pi
    thetas = np.r_[thetas, thetas[ltp] + np.pi, thetas[~ltp] - np.pi]
    phis   = np.r_[phis, np.pi - phis]

make_tpr(1000)

def init(vol, sigma, pr, only_seg=False, band=3,
         seed=None, spacing=[1.0, 1.0, 1.0], adjust_sigma=True,
         ball_small=True, min_vol=40.0, verbose=True):
    """
    Parameters
    ----------
        vol: ndarray
            The image volume.

        sigma: float, or ndarray len=3
            If single sigma, isotropic blur is peformed. Otherwise,
            a len 3 array or list is provided and anisotropic blurring 
            is performed.

        pr: float, range=[0,100]
            Radius percentile. Radial distances observed beyond the 
            `pr` percentile are removed.

        seed: list or ndarray, len=3
            The seed (possibly floating type) should be given in 
            index coordinates, i.e., it should *not* take into account 
            axis spacing.

        spacing: list or ndarray, len=3
            The spacings along each axis.

        adjust_sigma: bool, default=True
            Adjust the provided `sigma` value(s) by taking into account the
            provided `spacing` values.

        ball_small: bool, default=True
            If the volume of the initialization is less than `min_vol`
            then the initialization is simply set to a ball of volume
            `min_vol`.

        min_vol: float, default=40.0
            See `ball_small` parameter.

        verbose: bool, default=True
            If `ball_small` is True and a volume less than `min_vol` is
            observed *and* `verbose` is True, then an error message is 
            printed.
    """
    global thetas, phis, ntpr, radii
    assert ntpr is not None

    spacing = np.array(spacing)
    
    try:
        sigma = float(sigma)
        sigma = np.ones(3, dtype=np.float)*sigma
    except TypeError:
        if not isinstance(sigma, list) and not isinstance(sigma, np.ndarray):
            raise TypeError
        if len(sigma) != 3:
            raise ValueError

    if adjust_sigma:
        sigma = sigma.copy()
        sigma /= spacing

    # Local thresholding
    G = vol.copy()
    for i in range(3):
        G = gf1d(G, sigma[i], axis=i)
    B = (vol >= G)

    # Compute radii and determine radius threshold.
    rfm.radii_from_mask(thetas=thetas, phis=phis, seed=seed*spacing, mask=B,
                        radii=radii, di=spacing[0], dj=spacing[1],
                        dk=spacing[2])
    rad_thresh = np.percentile(radii, pr)

    if return_R: return radii

    # Set voxels outside of radius threshold to zero.
    ts = [np.arange(vol.shape[i], dtype=np.float)*spacing[i]
             for i in range(3)]
    x,y,z = np.meshgrid(*ts, indexing='ij')
    dist = ((x-seed[0]*spacing[0])**2 + 
            (y-seed[1]*spacing[1])**2 + 
            (z-seed[2]*spacing[2])**2)**0.5
    B[dist > rad_thresh] = False

    # Remove non-connected components that might arise from radius clipping.
    L,num = label(B, return_num=True)
    if num > 1:
        B = (L == L[tuple(seed.round().astype(np.int))])

    # If B is too small, initialize to a ball of volume ~ MINVOL.
    if ball_small and B.sum()*np.prod(spacing) < min_vol:
        B = np.zeros(vol.shape, dtype=np.bool)
        rad = (min_vol / (4/3.) / np.pi)**(1./3)
        B[dist <= rad] = True
        if verbose:
            sys.stderr.write("init < min_vol. Using ball init.\n")
            sys.stderr.flush()

    if only_seg: return B

    # Now we compute u0, the intial values of the level set
    # as +/-1 where B is 1/0. We also compute the distance transform
    # to obtain the narrow band mask.
    u0 = B.astype(np.float)
    u0 *= 2.0; u0 -= 1.0;

    if band > 0:
        dist = skfmm.distance(u0, dx=spacing, narrow=band)
        try:
            mask = ~dist.mask
            dist = dist.data
        except:
            # narrow band resulted in entire seg?
            if dist.flatten()[0] > 0:
                mask = np.ones(u0.shape, dtype=np.bool)
            else:
                mask = np.zeros(u0.shape, dtype=np.bool)
    else:
        mask = np.ones(u0.shape, dtype=np.bool)

    if band > 0:
        return u0, dist, mask
    else:
        return u0, mask
