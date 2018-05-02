"""
The module is a wrapper for the f2py wrapper. It's purely
a convenience wrapper. For more precision, use the `_masked_grad`
module directly, or tweak the C code!

-Matt Hancock, 2018

"""
import numpy as np
import _masked_grad

def gradient_centered(A, mask=None, dx=None, return_gmag=True, 
                      normalize=False):
    """
    Compute the centered difference approximations of the partial 
    derivatives of `A` along each coordinate axis, computed only 
    where `mask` is true.

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

    return_gmag: bool, default=True
        If True, the gradient magnitude is computed and returned also.

    normalize: bool, default=False
        If True, then the gradient terms are normalized so that the 
        gradient magnitude is one if computed over the gradient terms.
        Note that if `return_mag` is True, the gradient magnitude is 
        magnitude value prior to normalization (not necessarily one).

    Returns
    -------
    Gradients, Gradient Magnitude : [D1, ..., Dn], gmag 
        Returns the gradient along each axis approximated by centered
        differences (only computed where mask is True). The gradient magnitude 
        is optionally returned.
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

    D = [np.zeros_like(A) for _ in range(ndim)]
    gmag = np.zeros_like(A)

    # Select the correct function depending on dimension of the input.
    func = getattr(_masked_grad, 'gradient_centered%dd' % ndim)

    if ndim == 3:
        func(A=A, di=D[0], dj=D[1], dk=D[2], gmag=gmag, mask=mask,
             deli=dx[0], delj=dx[1], delk=dx[2],
             normalize=(1 if normalize else 0))
    elif ndim == 2:
        func(A=A, di=D[0], dj=D[1], gmag=gmag, mask=mask,
             deli=dx[0], delj=dx[1],
             normalize=(1 if normalize else 0))
    elif ndim == 1:
        func(A=A, di=D[0], gmag=gmag, mask=mask,
             deli=dx[0],
             normalize=(1 if normalize else 0))

    if return_gmag:
        return D, gmag
    else:
        return gmag

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
    Gradient Magnitude :  gmag 
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
