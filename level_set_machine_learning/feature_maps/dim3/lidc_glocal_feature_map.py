import numpy as np
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.spatial.distance import pdist

from level_set_machine_learning.feature_maps import feature_map_base
from level_set_machine_learning.feature_maps.dim3.utils import normal_samples as ns
from level_set_machine_learning.feature_maps.dim3.utils import to_com_samples as ts
from level_set_machine_learning.utils import masked_grad as mg

class lidc_glocal_feature_map(feature_map_base):
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

    nglocalimg = 4 # (2*2) normal samples, com samples [in/out]
    nglocalseg = 6 # slice*3, (slice diff)*3

    nglobalimg = 3  # mean, std, edge
    nglobalseg = 12 # area, len, iso, moments*2*3, (dist to com stats)*3

    def __init__(self, sigmas=[0, 3], nglocal_samples=10):
        self.sigmas = sigmas
        self.nglocal_samples = nglocal_samples
        self.nglocalimg *= nglocal_samples
        imgtotal = self.nlocalimg + self.nglocalimg + self.nglobalimg
        segtotal = self.nlocalseg + self.nglocalseg + self.nglobalseg
        self.nfeatures = imgtotal*len(sigmas) + segtotal

    def __call__(self, u, img, dist, mask, dx=None, memoize=None):
        assert u.ndim == 3 and img.ndim == 3 and \
               dist.ndim == 3 and mask.ndim == 3
        assert memoize in [None, 'create', 'use']

        shp = u.shape
        nmask = mask.sum()

        # TODO: add memoization here to re-use `samples` as a container
        samples = np.zeros(shp + (self.nglocal_samples, 2))

        ddi,ddj,ddk = mg.gradient_centered(dist, mask=mask, dx=dx,
                                           normalize=True, return_gmag=False)

        dx = np.ones(img.ndim) if dx is None else dx

        F = np.zeros(img.shape + (self.nfeatures,))

        pdx = np.prod(dx)
        H = (u > 0)
        Hf = H.astype(np.float)

        # Volume
        V = H.sum() * pdx
        F[mask,0] = V

        dHi,dHj,dHk = np.gradient(Hf, *dx)
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
        F[mask,3] = (ii*Hf / V).sum() * pdx
        F[mask,4] = (jj*Hf / V).sum() * pdx
        F[mask,5] = (kk*Hf / V).sum() * pdx

        com = np.array([F[mask,3][0], F[mask,4][0], F[mask,5][0]])
        com = np.array([(i*Hf).sum() / V for i in np.indices(H.shape,
                                                             dtype=np.float)])

        # Second moments.
        F[mask,6] = ((ii**2)*Hf / V).sum() * pdx
        F[mask,7] = ((jj**2)*Hf / V).sum() * pdx
        F[mask,8] = ((jj**2)*Hf / V).sum() * pdx

        pts = np.array(np.where((gmagH > 0) & mask)).T
        cdists = pdist(pts)

        # Stats of distances to center-of-mass distances along zero level set.
        F[mask, 9] = cdists.mean()
        F[mask,10] = cdists.std()
        F[mask,11] = cdists.max()

        # Distance from center of mass.
        F[mask,12] = np.sqrt((ii[mask]-F[mask,3][0])**2 + \
                             (jj[mask]-F[mask,4][0])**2 + \
                             (kk[mask]-F[mask,5][0])**2)

        # Area sums over first axis.
        s = np.apply_over_axes(np.sum, Hf, [1,2])
        F[:,:,:,13] = s
        s = np.abs(np.gradient(s, axis=0))
        F[:,:,:,14] = s

        # Area sums over second axis.
        s = np.apply_over_axes(np.sum, Hf, [0,2])
        F[:,:,:,15] = s
        s = np.abs(np.gradient(s, axis=1))
        F[:,:,:,16] = s

        # Area sums over third axis.
        s = np.apply_over_axes(np.sum, Hf, [0,1])
        F[:,:,:,17] = s
        s = np.abs(np.gradient(s, axis=2))
        F[:,:,:,18] = s

        for isig,sigma in enumerate(self.sigmas):
            ijump = self.nglobalseg + self.nglocalseg + self.nlocalseg + \
                    isig*(self.nlocalimg + self.nglocalimg + self.nglobalimg)

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
            F[mask,ijump+2] = blur[H].mean() # Global image mean.
            F[mask,ijump+3] = blur[H].std()  # Global image std.
            F[mask,ijump+4] = (gmag*gmagH).sum() / S # Global image edge.
            
            # Compute and store the normal ray samples.
            ns.get_samples(img=blur, com=com, nsamples=self.nglocal_samples,
                           ni=ddi, nj=ddj, nk=ddk,
                           di=dx[0], dj=dx[1], dk=dx[2],
                           mask=mask, samples=samples)
            sample_vals = samples[mask].reshape(nmask, 2*self.nglocal_samples)

            start = ijump+5
            stop  = ijump+5+2*self.nglocal_samples
            F[mask,start:stop] = sample_vals

            # Compute and store the center of mass ray samples.
            ts.get_samples(img=blur, com=com, nsamples=self.nglocal_samples,
                           di=dx[0], dj=dx[1], dk=dx[2],
                           mask=mask, samples=samples)
            sample_vals = samples[mask].reshape(nmask, 2*self.nglocal_samples)

            start = ijump+5+2*self.nglocal_samples
            stop  = ijump+5+2*self.nglocal_samples+2*self.nglocal_samples
            F[mask,start:stop] = sample_vals

        return F

    @property
    def names(self):
        """
        Return the list of feature names.
        """
        feats = ['volume', 'surface area', 'isoperimetric',
                 'moment 1 axis i', 'moment 1 axis j', 'moment 1 axis k',
                 'moment 2 axis i', 'moment 2 axis j', 'moment 2 axis k',
                 'com dist mean', 'com dist std', 'com dist max',
                 'dist from center of mass',
                 'slice area axis i', 'slice area diff axis i',
                 'slice area axis j', 'slice area diff axis j',
                 'slice area axis k', 'slice area diff axis k']

        img_feats = ['local img val', 'local img edge', 
                     'glob img avg', 'glob img std', 'glob edge']

        for i in range(self.nglocal_samples):
            img_feats += ['img normal(+)%d'%(i+1)]
            img_feats += ['img normal(-)%d'%(i+1)]
        for i in range(self.nglocal_samples):
            img_feats += ['img to com(+)%d'%(i+1)]
            img_feats += ['img to com(-)%d'%(i+1)]

        for s in self.sigmas:
            feats.extend(["%s-sigma=%.1f" % (f,s) for f in img_feats])

        return feats
