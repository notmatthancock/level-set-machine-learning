import numpy


def balance_mask(arr, random_state):
    """ Returns a mask that randomly partitions `arr` to have equal
    negative and positive parts

    Parameters
    ----------
    arr: ndarray
        The array of real values

    random_state: numpy.random.RandomState
        A random state object to randomly partition the data

    Returns
    -------
    mask: ndarray
        A mask where `arr[mask]` has equal parts negative and positive

    Note
    ----
    If `arr` is not able to be balanced (i.e., it is all positive,
    all negative, or all zeros), then mask will be all ones.
    """
    n_pos = (arr > 0).sum()
    n_neg = (arr < 0).sum()

    if (arr >= 0).all() or (arr <= 0).all():
        return numpy.ones(arr.shape, dtype=numpy.bool)
    elif n_pos > n_neg:  # then down-sample the positive elements
        where_pos = numpy.where(arr > 0)[0]
        indices = random_state.choice(where_pos, replace=False, size=n_neg)
        mask = arr <= 0
        mask[indices] = True
    elif n_pos < n_neg:  # then down-sample the negative elements
        where_neg = numpy.where(arr < 0)[0]
        indices = random_state.choice(where_neg, replace=False, size=n_pos)
        mask = arr >= 0
        mask[indices] = True
    else:
        mask = numpy.ones(arr.shape, dtype=numpy.bool)

    return mask
