import numpy as np
import skfmm

def sample_from_distance_map(h5file, seeds_per_image=10,
                             p=1.0, rs=None, verbose=True):
    """
    Generate seeds for segmentation by sampling the distance map.

    Parameters
    ----------
    h5file: h5py.File
        An appropriately formatted h5py segmentation dataset.

    seeds_per_image: int
        The number of seeds to generate per image.

    rs: np.random.RandomState, default=None
        A random state object for reproducibility

    Returns
    -------
    seeds: dict
        For each key in `h5file`, `seeds[key]` is a list of tuples,
        where each tuple is a seed for the corresponding image indexed
        by `key`.
    
    """
    rs = np.random.RandomState() if rs is None else rs

    seeds = {}

    total = len(h5file)
    pstr = "%%0%dd / %d" % (len(str(total)), total)

    for ikey, key in enumerate(h5file):
        if verbose: print(pstr % (ikey+1))

        seg = h5file[key+'/seg'][...]
        dx = h5file[key].attrs['dx']
        dist = h5file[key+'/dist'][...]

        # Select only points in seg (and convert to flat array).
        prob = dist[seg]
        # Clean up.
        prob[prob < 0] = 0
        # Normalize.
        prob /= prob.sum()

        # Indices where the segmentation is True.
        seg_inds = np.array(np.where(seg)).T

        # Total number of points in seg that are True.
        N = seg_inds.shape[0] # == seg.sum()

        # Subsample of indices into `seg_inds`
        seed_inds = rs.choice(N, size=seeds_per_image, replace=True, p=prob)

        seeds[key] = seg_inds[seed_inds,:]

        if ikey == 10: break

    return seeds
