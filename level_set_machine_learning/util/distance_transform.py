import numpy
import skfmm


def distance_transform(arr, band, dx):
    """ A thin wrapper around the skfmm distance transform function, but
    handles edge cases where the supplied array is completely negative or
    positive.

    Parameters
    ----------
    arr: numpy.ndarray
        The array on which the distance transform is to be computed

    band: float
        The narrow band parameter

    dx: numpy.ndarray, shape=arr.shape
        The delta terms

    Returns
    -------
    dist, mask: numpy.ndarray (dtype=float), numpy.ndarray (dtype=bool)
        The signed distance transform of `arr` and a boolean field `mask`
        that indicates a distance `band` from the zero level set.

    Note
    ----
    In the case that `arr` is zeros everywhere, then the returned `mask` will
    be True every where and the returned `dist` will be zero everywhere.
    In the case that the mask is everywhere positive or everywhere negative,
    then the returned mask will be False everywhere and the returned distance
    matrix will be +/- numpy.inf.
    """

    if (arr == 0).all():
        dist = numpy.zeros_like(arr)
        mask = numpy.ones(arr.shape, dtype=numpy.bool)
        return dist, mask

    # Check for the every positive or negative cases
    n_pos = (arr > 0).sum()
    if n_pos == arr.size or n_pos == 0:
        mask = numpy.zeros(arr.shape, dtype=numpy.bool)
        sign = numpy.sign(arr.ravel()[0])
        dist = sign * numpy.full(arr.shape, numpy.inf)
        return dist, mask

    dist = skfmm.distance(arr, narrow=band, dx=dx)

    if hasattr(dist, 'mask'):
        mask = ~dist.mask
        dist = dist.data
    else:
        # If no mask, then the band was large enough to
        # include the entire domain.
        mask = numpy.ones(arr.shape, dtype=numpy.bool)

    return dist, mask
