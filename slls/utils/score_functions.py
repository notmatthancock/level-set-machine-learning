import numpy as np


def jaccard(u, seg, t=0.0):
    """ Compute the Jaccard overlap score between `u > t` and `seg`. Also
    known as the intersection over union.
    """
    if seg.dtype != np.bool:
        msg = "seg dtype ({}) was not of type bool"
        raise ValueError(msg.format(seg.dtype))

    thresholded = u > t
    intersection = (thresholded & seg).sum() * 1.0
    union = (thresholded | seg).sum() * 1.0

    if intersection == 0:
        return 1.0
    else:
        return intersection / union
