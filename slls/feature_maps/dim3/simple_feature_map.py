import numpy as np
from slls.feature_maps import feature_map_base
from scipy.ndimage import gaussian_filter1d as gf1d
from slls.utils import masked_grad as mg

class simple_feature_map(feature_map_base):
    """
    A simple feature map with local and global, shape and image features,
    with image features computed at multiple scales.

    The feature map computation is implemented in the `__call__` member
    function so that the object can be called like in the example below.

    Example::

        TODO

    Documentation for `__call__`:

    Parameters
    ----------
    u: ndarrary
        The "level set function".

    img: ndarray
        The image.

    dist: ndarray
        The signed distance transform to the level set `u` (only computed
        necessarily in the narrow band region).

    mask: ndarray, dtype=bool
        The boolean mask indicating the narrow band region of `u`.

    dx: ndarray, shape=img.ndim
        The "delta" spacing terms for each image axis. If None, then
        1.0 is used for each axis.

    memoize: flag, default=None
        Should be one of [None, 'create', 'use']. This variable will be None
        always during training, but during run-time (where the feature 
        map is called for many iterations on the same image), it is
        'create' at the first iteration and 'use' for all iterations 
        thereafter. Use this to create more efficient run-time 
        performance by storing features that can be re-used
        in subsequent iterations.

    """
    nlocalimg  = 2 # img, grad
    nlocalseg  = 1 # dist from com
    nglobalimg = 2 # mean, std
    nglobalseg = 9 # area, len, iso, mi1, mj1, mk1, mi2, mj2, mk2

    def __init__(self, sigmas=[0, 3]):
        self.sigmas = sigmas
        self.nfeatures = (self.nlocalimg + self.nglobalimg)*len(sigmas) + \
                          self.nlocalseg + self.nglobalseg

    def __call__(self, u, img, dist, mask, dx=None, memoize=None):
        assert u.ndim == 3 and img.ndim == 3 and \
               dist.ndim == 3 and mask.ndim == 3
        assert memoize in [None, 'create', 'use']

        dx = np.ones(img.ndim) if dx is None else dx

        F = np.zeros(img.shape + (self.nfeatures,))

        pdx = np.prod(dx)
        H = (u > 0)

        # Volume
        V = H.sum() * pdx
        F[mask,0] = V

        dHi,dHj,dHk = np.gradient(H.astype(np.float), *dx)
        gmagH = np.sqrt(dHi**2 + dHj**2 + dHk**2)

        # Surface area
        # This uses a volume integral:
        # :math:`\\int\\int \\delta(u) \\| Du \\| dx dy`
        # to compute the length of the surface via Co-Area formula.
        S = gmagH.sum() * pdx
        F[mask,1] = S

        # Isoperimetric ratio
        F[mask,2] = 36*np.pi*V**2 / S**3
        
        if memoize is None:
            ii,jj,kk = np.indices(img.shape, dtype=np.float)
            ii *= dx[0]; jj *= dx[1]; kk *= dx[2]
        else:
            if memoize == 'create':
                self.ii, self.jj, self.kk = np.indices(img.shape,
                                                       dtype=np.float)
                self.ii *= dx[0]; self.jj *= dx[1]; self.kk *= dx[2]
            # Handles both 'create' and 'use' cases.
            ii,jj,kk = self.ii, self.jj, self.kk

        # First moments.
        F[mask,3] = (ii*H.astype(np.float) / V).sum() * pdx
        F[mask,4] = (jj*H.astype(np.float) / V).sum() * pdx
        F[mask,5] = (kk*H.astype(np.float) / V).sum() * pdx

        # Second moments.
        F[mask,6] = ((ii**2)*H.astype(np.float) / V).sum() * pdx
        F[mask,7] = ((jj**2)*H.astype(np.float) / V).sum() * pdx
        F[mask,8] = ((jj**2)*H.astype(np.float) / V).sum() * pdx

        # Distance from center of mass.
        F[mask,9] = np.sqrt((ii[mask]-F[mask,3][0])**2 + \
                            (jj[mask]-F[mask,4][0])**2 + \
                            (kk[mask]-F[mask,5][0])**2)

        for isig,sigma in enumerate(self.sigmas):
            ijump = self.nglobalseg + self.nlocalseg + \
                    isig*(self.nlocalimg + self.nglobalimg)

            if memoize is None:
                if sigma > 0:
                    blur = img
                    for axis in range(len(dx)):
                        blur = gf1d(blur, sigma/dx[axis], axis=axis)
                    _,gmag = mg.gradient_centered(blur, mask=mask, dx=dx) 
                else:
                    blur = img
                    _,gmag = mg.gradient_centered(blur, mask=mask, dx=dx) 
            else:
                if memoize == 'create':
                    if not hasattr(self, 'blurs'):
                        self.blurs = np.zeros((len(self.sigmas),) + img.shape)
                    if not hasattr(self, 'gmags'):
                        self.gmags = np.zeros((len(self.sigmas),) + img.shape)

                    mask_all = np.ones(img.shape, dtype=np.bool)

                    if sigma > 0:
                        self.blurs[isig] = img
                        for axis in range(len(dx)):
                            self.blurs[isig] = gf1d(self.blurs[isig],
                                                    sigma/dx[axis],
                                                    axis=axis)
                        _, self.gmags[isig] = mg.gradient_centered(
                                                self.blurs[isig],
                                                mask=mask_all,
                                                dx=dx
                                              )
                    else:
                        self.blurs[isig] = img
                        _, self.gmags[isig] = mg.gradient_centered(
                                                self.blurs[isig],
                                                mask=mask_all,
                                                dx=dx
                                              )

                # Handles both 'create' and 'use' `memoize` cases.
                blur = self.blurs[isig]
                gmag = self.gmags[isig]

            F[mask,ijump+0] = blur[mask]     # Local image.
            F[mask,ijump+1] = gmag[mask]     # Local image edge.
            F[mask,ijump+2] = blur[H].mean() # Global image.
            F[mask,ijump+3] = blur[H].std()  # Global image edge.

        return F

    @property
    def names(self):
        """
        Return the list of feature names.
        """
        feats = ['volume', 'surface area', 'isoperimetric',
                 'moment 1 axis i', 'moment 1 axis j', 'moment 1 axis k',
                 'moment 2 axis i', 'moment 2 axis j', 'moment 2 axis k',
                 'dist from center of mass']

        img_feats = ['img-val', 'img-edge', 'img-avg', 'img-std']
        for s in self.sigmas:
            feats.extend(["%s-sigma=%.1f" % (f,s) for f in img_feats])

        return feats
