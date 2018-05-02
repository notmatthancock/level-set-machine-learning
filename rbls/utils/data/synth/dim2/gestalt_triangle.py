import sys
import numpy as np
from scipy.stats import beta
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as mpath

# s = 2 sqrt(3) r
# h = sqrt(3)/2 s
# h = 3r

def make(n, side_len, circ_rad, theta, ishift=0, jshift=0,
         sigma_smooth=0., sigma_noise=0., rs=None):
    """
    Create an illusory triangle contour [1] image with random
    size and orientation.

    [1]: https://en.wikipedia.org/wiki/Illusory_contours

    Parameters
    ----------
    n: int
        Image shape will be (n,n)

    side_len: float
        Side length of the triangle in pixels.

    circ_rad: float
        Radius of the circles at the vertices
        of the triangle in pixels.

    theta: float (radians)
        Rotation of the triangle. Zero points the triangle to the right.

    ishift,jshift: integers
        Translate the center of the triangle by ishift and jshift.

    sigma_smooth: float
        Gaussian smoothing parameter (make image borders more diffuse).

    sigma_noise: float
        Additive noise amplitude.

    rs: numpy.random.RandomState, default=None
        Include for reproducible results.
    """
    if circ_rad > 0.5*side_len:
        raise ValueError(("Circle radius should be less "
                          "than one half the side length."))

    # Triangle height.
    height = 0.5*np.sqrt(3)*side_len

    # Distance from center of triangle to a vertex.
    tri_rad = (2.0/3.0)*height

    # Rotation factor for triangle vertices.
    w = (2.0/3.0)*np.pi

    # Get extent of triangle plus outer circles for validation.
    extent = np.zeros((3,2))
    for i in range(3):
        x = (tri_rad+circ_rad)*np.cos(i*w+theta) + n/2 + jshift
        y = (tri_rad+circ_rad)*np.sin(i*w+theta) + n/2 - ishift
        extent[i] = n-y,x

    for e in extent:
        if e[0] < 0 or e[0] > n-1:
            raise ValueError(("Extent of triangle plus circles exceeds"
                              "image dimensions along axis 0."))
        if e[1] < 0 or e[1] > n-1:
            raise ValueError(("Extent of triangle plus circles exceeds"
                              "image dimensions along axis 1."))

    vertices = np.zeros((3,2))
    for i in range(3):
        x = tri_rad*np.cos(i*w+theta) + n/2 + jshift
        y = tri_rad*np.sin(i*w+theta) + n/2 - ishift
        vertices[i] = n-y,x

    tri_path = mpath(np.append(vertices, vertices[-1].reshape(1,2), axis=0),
                     codes=[mpath.MOVETO, mpath.LINETO,
                            mpath.LINETO, mpath.CLOSEPOLY])

    ii,jj = np.indices((n,n))
    coords = np.c_[ii.flatten(), jj.flatten()]

    triangle = tri_path.contains_points(coords).reshape(n,n)

    ucircle = mpath.unit_circle()
    circles = np.zeros((n,n), dtype=np.bool)

    for v in vertices:
        circle = mpath(vertices=ucircle.vertices*circ_rad + v,
                       codes=ucircle.codes)
        circles = np.logical_or(circles,
                                circle.contains_points(coords).reshape(n,n))
    
    image = (~np.logical_and(circles, ~triangle)).astype(np.float)
    rs = rs if rs is not None else np.random.RandomState()

    if sigma_smooth > 0:
        image = gaussian_filter(image, sigma_smooth)

    if sigma_noise > 0:
        image += sigma_noise*rs.randn(n,n)

    return image, triangle

def make_dataset(N, n=101, slen=[40,60], crad=[10,20],
                 shift=[-0, 0], nsig=[0.05,0.15], ssig=[1,1],
                 theta=[0,2*np.pi/3], rs=None, verbose=True):
    """
    Make a randomly generated dataset of illusory triangle data.

    Parameters
    ----------
    N: int
        The number of examples.

    n: int
        The image size.

    slen: list, len=2
        Interval of triangle side lengths from which to sample.

    crad: list, len=2
        Interval of circle radii from which to sample.

    shift: list, len=2
        The interval of shift values from which to sample.

    nsig: list, len=2
        The interval of values from which to sample `sigma_noise`. 

    ssig: list, len=2
        The interval of values from which to sample `sigma_smooth`. 

    ctheta: list, len=2
        The interval of values form which to sample `theta`.

    return_meta: bool, default=False
        Return a list of meta data attributes for each example if True.

    rs: numpy.random.RandomState, default=None
        Include a for reproducible results.

    verbose: bool, default=True
        Print progress.
    """
    rs = rs if rs is not None else np.random.RandomState()
    def betarvs(**kwargs):
        return beta.rvs(3,3,random_state=rs,**kwargs)

    if verbose:
        q = len(str(N))
        pstr = "Creating dataset ... %%0%dd / %d" % (q,N)

    imgs = np.zeros((N, n, n))
    segs = np.zeros((N, n, n), dtype=np.bool)

    i = 0

    while i < N:
        try:
            sl = betarvs(loc=slen[0],scale=slen[1]-slen[0])
            cr = betarvs(loc=crad[0],scale=crad[1]-crad[0])

            ishift,jshift = betarvs(loc=shift[0],
                                     scale=shift[1]-shift[0], size=2)
            th = betarvs(loc=theta[0],scale=theta[1]-theta[0])

            sigma_noise  = betarvs(loc=nsig[0], scale=nsig[1]-nsig[0])
            sigma_smooth = betarvs(loc=ssig[0], scale=ssig[1]-ssig[0])

            meta = dict(
                side_len=sl,
                circ_rad=cr,
                ishift=ishift,
                jshift=jshift,
                theta=th,
                sigma_smooth=sigma_smooth,
                sigma_noise=sigma_noise
            )

            img,seg = make(n, sl, cr, th, ishift, jshift,
                           sigma_smooth, sigma_noise, rs=rs)
            imgs[i] = img
            segs[i] = seg
            i+=1

            if verbose: print(pstr % i)
        except ValueError:
            continue
    return imgs, segs
