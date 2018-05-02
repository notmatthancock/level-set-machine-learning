import sys
import numpy as np
from scipy.stats import beta
from scipy.ndimage import gaussian_filter


def make(n=71, r=15, ishift=0, jshift=0, kshift=0,
         sigma_noise=0.1, sigma_smooth=2, 
         plane_b=0, plane_n=np.ones(3)/np.sqrt(3), plane_thickness=5,
         rs=None):
    """
    Make an image volume of a bright sphere with a slab removed. The 
    corresponding ground-truth segmentation is the sphere of minimal 
    radius enclosing the bright regions.

    Parameters
    ----------
    n: int, default=71
        The image and segmentation arrays will be shape (n,n).

    r: float, default=15
        Radius of the circle.

    ishift,jshift,kshift: int, default=0
        The center of the circle (index offsets from the center of the image).

    sigma_noise: float, default=0.1
        The additive noise amplitude to be added to the image.

    sigma_smooth: float, default=2
        The Gaussian smoothing factor to apply to the image.

    plane_b: float, default=0
        The offset parameter of the slab through the sphere.

    plane_n: float
        The orientation (normal) vector of the slab through the sphere.

    plane_thickness: float, default=5
        The region a distance `cut_thickness` from the line through
        the circle will be set to the background image value.

    rs: numpy.Random.RandomState
        RandomState object for reproducible results.

    Returns
    -------
    img, seg, meta : ndarray (dtype=float), ndarray (dtype=bool), dict
        The image and segmentation image are returned as well as 
        a dictionary of the parameters used.
    """
    ci = n/2 + ishift; cj = n/2 + jshift; ck = n/2 + kshift
    center = np.r_[ci,cj,ck]

    if not (ci+r < n-1 and ci-r > 0):
        raise ValueError("Circle outside bounds, axis 0.")
    if not (cj+r < n-1 and cj-r > 0):
        raise ValueError("Circle outside bounds, axis 1.")
    if not (ck+r < n-1 and ck-r > 0):
        raise ValueError("Circle outside bounds, axis 2.")
    if plane_b + plane_thickness > r:
        raise ValueError("`plane_b + plane_thickness` should be less than radius.")

    rs = rs if rs is not None else np.random.RandomState()
    ii,jj,kk = np.indices((n,n,n), dtype=np.float) - center.reshape(3,1,1,1)

    seg = np.sqrt(ii**2 + jj**2 + kk**2) <= r

    # Remove plane from sphere to create image volume.
    dist = np.abs(plane_n[0]*ii + plane_n[1]*jj + plane_n[2]*kk + plane_b)
    cut = dist <= plane_thickness
    img = np.logical_and(seg, ~cut).astype(np.float)

    img = gaussian_filter(img, sigma_smooth)
    img *= (~cut).astype(np.float)
    img = (img - img.min()) / (img.max() - img.min())
    img += sigma_noise*rs.randn(n,n,n)

    return img,seg


def make_dataset(N, n=71, rad=[15,25], shift=[0,0],
                 nsig=[0.1,0.4], ssig=[0,1.], pthick=[3,7],
                 pb=[0,5], rs=None, verbose=True):
    """
    Make a randomly generated dataset of hamburger data.

    Parameters
    ----------
    N: int
        The number of examples.

    n: int
        The image size.

    rad: list, len=2
        Interval of radii from which to sample.

    shift: list, len=2
        The interval of shift values from which to sample.

    nsig: list, len=2
        The interval of values from which to sample `sigma_noise`. 

    ssig: list, len=2
        The interval of values from which to sample `sigma_smooth`. 

    pthick: list, len=2
        The interval of values from which to sample `plane_thickness`.

    pb: list, len=2
        The interval of values form which to sample `plane_b`.

    rs: numpy.random.RandomState, default=None
        Include a for reproducible results.
    """
    rs = rs if rs is not None else np.random.RandomState()
    def betarvs(**kwargs):
        return beta.rvs(3,3,random_state=rs,**kwargs)

    imgs = np.zeros((N,n,n,n))
    segs = np.zeros((N,n,n,n), dtype=np.bool)

    if verbose:
        q = len(str(N))
        pstr = "Creating dataset ... %%0%dd / %d" % (q,N)

    i = 0

    while i < N:
        try:
            r = betarvs(loc=rad[0],scale=rad[1]-rad[0])

            ishift,jshift,kshift = betarvs(loc=shift[0],
                                           scale=shift[1]-shift[0], size=3)
            sigma_noise  = betarvs(loc=nsig[0], scale=nsig[1]-nsig[0])
            sigma_smooth = betarvs(loc=ssig[0], scale=ssig[1]-ssig[0])
            plane_b = betarvs(loc=pb[0], scale=pb[1]-pb[0])
            plane_thickness = betarvs(loc=pthick[0], scale=pthick[1]-pthick[0])
            plane_n = rs.randn(3); plane_n /= np.linalg.norm(plane_n)
            img,seg = make(n, r, ishift, jshift, kshift,
                           sigma_noise, sigma_smooth,
                           plane_b, plane_n, plane_thickness, rs)
            imgs[i] = img
            segs[i] = seg
            i+=1
            
            if verbose: print(pstr % i)
        except ValueError:
            continue

    return imgs, segs 
