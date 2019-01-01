import ctypes

import numpy as np
from numpy.ctypeslib import ndpointer


_masked_gradient = ctypes.cdll.LoadLibrary('./_masked_gradient.so')



def gradient_centered(arr, mask=None, dx=None, return_gmag=True,
                      normalize=False):
    """
    Compute the centered difference approximations of the partial 
    derivatives of `arr` along each coordinate axis, computed only
    where `mask` is true.

    Note
    ----
    Only dimensions 1, 2, and 3 are supported.

    Parameters
    ----------
    arr: ndarray, dtype=float
        The gradient of `arr` is returned.

    mask: ndarray, dtype=bool, same shape as `arr`, default=None
        The gradient of `arr` is only computed where `mask` is true. If
        None (default), then mask is all ones.

    dx: ndarray, dtype=float, len=arr.ndim
        These indicate the "delta" or spacing terms along each axis.
        If None (default), then spacing is 1.0 along each axis.

    return_gmag: bool, default=True
        If True, the gradient magnitude is computed and returned also.

    normalize: bool, default=False
        If True, then the gradient terms are normalized so that the 
        gradient magnitude is one if computed over the gradient terms.
        Note that if `return_mag` is True, the gradient magnitude is 
        magnitude value prior to normalization (not necessarily one).

    Returns
    -------
    [D1, ..., Dn], gmag: list, ndarray
        Returns the gradient along each axis approximated by centered
        differences (only computed where mask is True). The gradient magnitude 
        is optionally returned.
    """
    ndim = arr.ndim
    assert 1 <= ndim <= 3, "Only dimensions 1-3 supported."
    if arr.dtype != np.float:
        raise ValueError("`arr` must be float type.")

    if mask is not None:
        if mask.ndim != ndim:
            raise ValueError("Shape mismatch between `mask` and `arr`.")
    else:
        mask = np.ones(arr.shape, dtype=np.bool)

    if dx is not None:
        if len(dx) != ndim:
            raise ValueError("`dx` vector shape mismatch.")
    else:
        dx = np.ones(ndim, dtype=np.float)

    D = [np.zeros_like(arr) for _ in range(ndim)]
    gmag = np.zeros_like(arr)

    # Select the correct function depending on dimension of the input.
    func = getattr(_masked_gradient, 'gradient_centered%dd' % ndim)

    if ndim == 3:
        func(A=arr, di=D[0], dj=D[1], dk=D[2], gmag=gmag, mask=mask,
             deli=dx[0], delj=dx[1], delk=dx[2],
             normalize=(1 if normalize else 0))
    elif ndim == 2:
        func(A=arr, di=D[0], dj=D[1], gmag=gmag, mask=mask,
             deli=dx[0], delj=dx[1],
             normalize=(1 if normalize else 0))
    elif ndim == 1:
        func(A=arr, di=D[0], gmag=gmag, mask=mask,
             deli=dx[0],
             normalize=(1 if normalize else 0))

    if return_gmag:
        return D, gmag
    else:
        return D


def gmag_os(A, nu, mask=None, dx=None):
    """
    This numerical approximation is an upwind approximation of 
    the velocity-dependent gradient magnitude term in the PDE:

    .. math::
        u_t = \\nu \\| Du \\|

    from Osher and Sethian [1]. This is why the function is called 
    `gmag_os`.

    In the PDE, the gradient vector of u is assumed to point inward and 
    :math:`\\nu` governs the velocity in the normal direction, with 
    positive values corresponding to movement in the outward normal
    direction (expansion) and negative values corresponding to the inward
    normal direction (contraction).

    Note that in Osher and Sethian's formulation they write:

    .. math::
        u_t + F \\| Du \\| = 0

    Thus, the relation is :math:`\\nu = -F`.

    [1]: Level Set Methods. Evolving Interfaces in Geometry, 
         Fluid Mechanics, Computer Vision, and Materials Science
         J.A. Sethian, Cambridge University Press, 1996
         Cambridge Monograph on Applied and Computational Mathematics

    Parameters
    ----------
    A: ndarray, dtype=float
        The gradient of `A` is returned.

    mask: ndarray, dtype=bool, same shape as `A`, default=None
        The gradient of `A` is only computed where `mask` is true. If
        None (default), then mask is all ones.

    dx: ndarray, dtype=float, len=A.ndim
        These indicate the "delta" or spacing terms along each axis.
        If None (default), then spacing is 1.0 along each axis.

    Returns
    -------
    gmag: ndarray
        The velocity-dependent gradient magnitude approximation.
    """
    ndim = A.ndim
    assert 1 <= ndim <= 3, "Only dimensions 1-3 supported."
    if A.dtype != np.float:
        raise ValueError("`A` must be float type.")

    if mask is not None:
        if mask.ndim != ndim:
            raise ValueError("Shape mismatch between `mask` and `A`.")
    else:
        mask = np.ones(A.shape, dtype=np.bool)

    if dx is not None:
        if len(dx) != ndim:
            raise ValueError("`dx` vector shape mismatch.")
    else:
        dx = np.ones(ndim, dtype=np.float)

    gmag = np.zeros_like(A)

    # Select the correct function depending on dimension of the input.
    func = getattr(_masked_grad, 'gmag_os%dd' % ndim)

    if ndim == 3:
        func(A=A, nu=nu, gmag=gmag, mask=mask,
             deli=dx[0], delj=dx[1], delk=dx[2])
    elif ndim == 2:
        func(A=A, nu=nu, gmag=gmag, mask=mask,
             deli=dx[0], delj=dx[1])
    elif ndim == 1:
        func(A=A, nu=nu, gmag=gmag, mask=mask,
             deli=dx[0])

    return gmag
