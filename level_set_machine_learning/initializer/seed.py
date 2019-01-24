
import numpy


def center_of_mass_seeder(dataset_example):
    """ Yields a seed as the computed center of mass of the segmentation

    Parameters
    ----------
    dataset_example: DatasetExample
        A dataset example instance from
        :class:`level_set_machine_learning.core.datasets_handler.DatasetExample`

    Returns
    -------
    seed: numpy.ndarray
        Array of seed points with length corresponding to the number of
        dimensions in the input.

    """
    seg = dataset_example.seg
    indices = numpy.indices(seg.shape, dtype=numpy.float)
    total = seg.sum()

    return [(ind*seg).sum() / total for ind in indices]
