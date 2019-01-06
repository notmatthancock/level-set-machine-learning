import numpy


def jaccard(u, seg, threshold=0.0):
    """ Compute the Jaccard overlap score between `u > threshold` and `seg`
    (Also known as the "intersection over union")
    """
    if seg.dtype != numpy.bool:
        msg = "`seg` dtype ({}) was not of type bool"
        raise ValueError(msg.format(seg.dtype))

    thresholded = u > threshold

    intersection = float((thresholded & seg).sum())
    union = float((thresholded | seg).sum())

    if union == 0:
        # If union is zero then both `seg` and `u > threshold` are empty! This
        # is an edge case, but we'll say that the "overlap" score is perfect,
        # rather than nil.
        return 1.0
    else:
        return intersection / union


def dice(u, seg, threshold=0.0):
    """ Compute the Dice coefficient between `u > threshold` and `seg`
    """
    jaccard_score = jaccard(u, seg, threshold)
    return 2.0 * jaccard_score / (jaccard_score + 1.0)
