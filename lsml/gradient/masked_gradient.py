import ctypes
import pkg_resources

import numpy as np
from numpy.ctypeslib import ndpointer


# Load the masked gradient C library
_masked_gradient = ctypes.cdll.LoadLibrary(
    pkg_resources.resource_filename(
        'lsml.util', '_cutil/masked_gradient.so'
    )
)


def _get_gradient_centered_func(ndim):
    """ Gets the function from the c module and sets up the respective
    argument and return types
    """
    func = getattr(_masked_gradient, 'gradient_centered{:d}d'.format(ndim))
    func.restype = None

    array_dimension_args = ((ctypes.c_int,) * ndim)
    array_arg = (ndpointer(ctypes.c_double),)
    mask_arg = (ndpointer(ctypes.c_bool),)
    gradient_args = (ndpointer(ctypes.c_double),) * ndim
    gradient_magnitude_arg = (ndpointer(ctypes.c_double),)
    delta_args = (ctypes.c_double,) * ndim
    normalize_arg = (ctypes.c_int,)

    func.argtypes = (
        array_dimension_args +
        array_arg +
        mask_arg +
        gradient_args +
        gradient_magnitude_arg +
        delta_args +
        normalize_arg
    )

    return func


def _get_gradient_magnitude_osher_sethian_func(ndim):
    """ Gets the function from the c module and sets up the respective
    argument and return types
    """
    func = getattr(_masked_gradient, 'gmag_os{:d}d'.format(ndim))
    func.restype = None

    array_dimension_args = ((ctypes.c_int,) * ndim)
    array_arg = (ndpointer(ctypes.c_double),)
    mask_arg = (ndpointer(ctypes.c_bool),)
    nu_arg = (ndpointer(ctypes.c_double),)
    gradient_magnitude_arg = (ndpointer(ctypes.c_double),)
    delta_args = (ctypes.c_double,) * ndim

    func.argtypes = (
        array_dimension_args +
        array_arg +
        mask_arg +
        nu_arg +
        gradient_magnitude_arg +
        delta_args
    )

    return func


def gradient_centered(arr, mask=None, dx=None,
                      return_gradient_magnitude=True,
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
        None (default), then mask True everywhere.

    dx: ndarray, dtype=float, len=arr.ndim
        These indicate the "delta" or spacing terms along each axis.
        If None (default), then spacing is 1.0 along each axis.

    return_gradient_magnitude: bool, default=True
        If True, the gradient magnitude is computed and returned also.

    normalize: bool, default=False
        If True, then the gradient terms are normalized so that the
        gradient magnitude is one if computed over the gradient terms.
        Note that if `return_mag` is True, the gradient magnitude is
        magnitude value prior to normalization (not necessarily one).

    Returns
    -------
    [gradient_1, ... , gradient_n], gradient_magnitude: list, ndarray
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

    gradients = [np.zeros_like(arr) for _ in range(ndim)]
    gradient_magnitude = np.zeros_like(arr)

    # Set up the C function
    func = _get_gradient_centered_func(ndim=ndim)

    # Set up the arguments to the C function
    args = (
        arr.shape +
        (arr,) +
        (mask,) +
        tuple(gradients) +
        (gradient_magnitude,) +
        tuple(dx) +
        (int(normalize),)
    )

    # Call the C function
    func(*args)

    if return_gradient_magnitude:
        return gradients, gradient_magnitude
    else:
        return gradients


def gradient_magnitude_osher_sethian(arr, nu, mask=None, dx=None):
    """
    This numerical approximation is an upwind approximation of
    the velocity-dependent gradient magnitude term in the PDE:

    .. math::
        u_t = \\nu \\| Du \\|

    from Osher and Sethian [1]. This is why the function is called
    `gradient_magnitude_osher_sethian`.

    In the PDE, the gradient vector of u is assumed to point inward and
    :math:`\\nu` governs the velocity in the normal direction, with
    positive values corresponding to movement in the outward normal
    direction (expansion) and negative values corresponding to the inward
    normal direction (contraction).

    Note that in Osher and Sethian's formulation they write:

    .. math::
        u_t + F \\| Du \\| = 0

    Thus, the correspondence is :math:`\\nu = -F`.

    [1]: Level Set Methods. Evolving Interfaces in Geometry,
         Fluid Mechanics, Computer Vision, and Materials Science
         J.A. Sethian, Cambridge University Press, 1996
         Cambridge Monograph on Applied and Computational Mathematics

    Parameters
    ----------
    arr: ndarray, dtype=float
        The gradient of `A` is returned.

    mask: ndarray, dtype=bool, same shape as `A`, default=None
        The gradient of `A` is only computed where `mask` is true. If
        None (default), then mask is all ones.

    dx: ndarray, dtype=float, len=A.ndim
        These indicate the "delta" or spacing terms along each axis.
        If None (default), then spacing is 1.0 along each axis.

    Returns
    -------
    gradient_magnitude: ndarray
        The velocity-dependent gradient magnitude approximation.
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

    gradient_magnitude = np.zeros_like(arr)

    # Set up the C function
    func = _get_gradient_magnitude_osher_sethian_func(ndim=ndim)

    # Set up the arguments to the C function
    args = (
        arr.shape +
        (arr,) +
        (mask,) +
        (nu,) +
        (gradient_magnitude,) +
        tuple(dx)
    )

    # Call the C function
    func(*args)

    return gradient_magnitude
