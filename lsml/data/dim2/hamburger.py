import logging

import numpy as np
from scipy.stats import beta
from scipy.ndimage import gaussian_filter


logger = logging.getLogger(__name__)


def make(n=101, r=25, ishift=0, jshift=0,
         sigma_noise=0.1, sigma_smooth=2, 
         cut_b=0, cut_theta=0, cut_thickness=5, rs=None):
    """
    Make a bright circle with a thick line removed.

    Parameters
    ----------
    n: int, default=101
        The image and segmentation arrays will be shape (n,n).

    r: float, default=25
        Radius of the circle.

    ishift,jshift: int, default=0
        The center of the circle (index offsets from the center of the image).

    sigma_noise: float, default=0.1
        The additive noise amplitude to be added to the image.

    sigma_smooth: float, default=2
        The Gaussian smoothing factor to apply to the image.

    cut_b: float, default=0
        The offset parameter of the line through the circle.

    cut_theta: float, default=0
        The orientation of the line through the circle.

    cut_thickness: float, default=5
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
    ci = n/2 + ishift; cj = n/2 + jshift

    if not (ci+r < n-1 and ci-r > 0):
        raise ValueError("Circle outside bounds, axis 0.")
    if not (cj+r < n-1 and cj-r > 0):
        raise ValueError("Circle outside bounds, axis 1.")
    if cut_b > r:
        raise ValueError("`cut_b` should be less than radius.")
    if cut_thickness > r:
        raise ValueError("`cut_thickness` should be less than radius.")

    rs = rs if rs is not None else np.random.RandomState()
    ii,jj = np.indices((n,n), dtype=np.float)

    seg = np.sqrt((ii-ci)**2 + (jj-cj)**2) <= r

    xx, yy = jj-cj, n-ii-1-(n-ci-1)

    # Take a cut out of the circle.
    dist = np.abs(np.cos(cut_theta)*xx + np.sin(cut_theta)*yy - cut_b)
    inline = dist <= cut_thickness
    img = np.logical_and(seg, ~inline).astype(np.float)

    img = gaussian_filter(img, sigma_smooth)
    img *= (~inline).astype(np.float)
    img = (img - img.min()) / (img.max() - img.min())
    img += sigma_noise*rs.randn(n,n)

    info = dict(
        ci=ci, cj=cj, r=r,
        sigma_noise=sigma_noise,
        sigma_smooth=sigma_smooth,
        cut_b=cut_b,
        cut_thickness=cut_thickness,
        cut_theta=cut_theta
    )

    return img,seg,info


def make_dataset(N, n=51, rad=[15,21], shift=[0,0],
                 nsig=[0.3,0.5], ssig=[0,0], cthick=[4,7],
                 ctheta=[0,2*np.pi], cb=[0,10], return_meta=False,
                 verbose=True, random_state=None, print_mistakes=False):
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

    cthick: list, len=2
        The interval of values from which to sample `cut_thickness`.

    ctheta: list, len=2
        The interval of values form which to sample `cut_theta`.

    cb: list, len=2
        The interval of values form which to sample `cut_b`.

    return_meta: bool, default=False
        Return a list of meta data attributes for each example if True.

    random_state: numpy.random.RandomState, default=None
        Include a for reproducible results.

    print_mistakes: bool, default=False
    """
    random_state = random_state if random_state is not None else np.random.RandomState()
    def betarvs(**kwargs):
        return beta.rvs(3, 3, random_state=random_state, **kwargs)

    imgs = np.zeros((N,n,n))
    segs = np.zeros((N,n,n), dtype=np.bool)

    if verbose:
        q = len(str(N))
        pstr = "Creating dataset ... %%0%dd / %d" % (q,N)

    if return_meta:
        meta = []

    i = 0

    while i < N :
        try:
            r = betarvs(loc=rad[0],scale=rad[1]-rad[0])

            ishift,jshift = beta.rvs(3, 3, loc=shift[0],
                                     scale=shift[1]-shift[0], size=2)
            sigma_noise  = betarvs(loc=nsig[0], scale=nsig[1]-nsig[0])
            sigma_smooth = betarvs(loc=ssig[0], scale=ssig[1]-ssig[0])
            cut_b = betarvs(loc=cb[0], scale=cb[1]-cb[0])
            cut_thickness = betarvs(loc=cthick[0], scale=cthick[1]-cthick[0])
            cut_theta = betarvs(loc=ctheta[0], scale=ctheta[1]-ctheta[0])
            img,seg,info = make(n, r, ishift, jshift,
                                sigma_noise, sigma_smooth,
                                cut_b=cut_b, cut_theta=cut_theta,
                                cut_thickness=cut_thickness, rs=random_state)
            imgs[i] = img
            segs[i] = seg
            if return_meta:
                meta.append(info)
            i+=1
            
            if verbose: logger.info(pstr % i)
        except ValueError as e:
            if print_mistakes:
                print("Bad params:", e, "... Continuing to next iteration")
            continue

    if return_meta:
        return imgs, segs, meta
    else:
        return imgs, segs 
